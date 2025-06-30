#!/usr/bin/env python3
"""
AALS Prometheus API クライアント
Prometheusからメトリクスデータを取得・分析
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, ConfigDict

from aals.core.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PrometheusMetric:
    """Prometheusメトリクスデータ"""
    metric_name: str
    labels: Dict[str, str]
    timestamp: datetime
    value: float
    
    def __str__(self) -> str:
        labels_str = ", ".join([f"{k}={v}" for k, v in self.labels.items()])
        return f"{self.metric_name}{{{labels_str}}} = {self.value}"


@dataclass 
class MetricRange:
    """メトリクス範囲データ"""
    metric_name: str
    labels: Dict[str, str]
    values: List[tuple[datetime, float]]
    
    @property
    def latest_value(self) -> Optional[float]:
        """最新値を取得"""
        return self.values[-1][1] if self.values else None
    
    @property
    def min_value(self) -> Optional[float]:
        """最小値を取得"""
        return min(v[1] for v in self.values) if self.values else None
    
    @property
    def max_value(self) -> Optional[float]:
        """最大値を取得"""
        return max(v[1] for v in self.values) if self.values else None
    
    @property
    def avg_value(self) -> Optional[float]:
        """平均値を取得"""
        if not self.values:
            return None
        return sum(v[1] for v in self.values) / len(self.values)


class PrometheusQueryResult(BaseModel):
    """Prometheusクエリ結果"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: str
    data: Dict[str, Any]
    error_type: Optional[str] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


class PrometheusAPIClient:
    """Prometheus API クライアント"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Args:
            base_url: PrometheusサーバーのベースURL
            timeout: HTTPタイムアウト（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info("Prometheus API Client initialized", 
                   base_url=base_url, timeout=timeout)
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        if self._client:
            await self._client.aclose()
    
    async def verify_connection(self) -> bool:
        """Prometheus接続確認"""
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            
            response = await self._client.get(f"{self.base_url}/api/v1/status/config")
            
            if response.status_code == 200:
                logger.info("Prometheus connection verified successfully")
                return True
            else:
                logger.warning("Prometheus connection failed", 
                             status_code=response.status_code,
                             response_text=response.text)
                return False
                
        except Exception as e:
            logger.error("Prometheus connection verification failed", 
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    async def query_instant(self, query: str, time: Optional[datetime] = None) -> PrometheusQueryResult:
        """
        即座クエリ実行
        
        Args:
            query: PromQLクエリ
            time: 指定時刻（Noneの場合は現在時刻）
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        params = {"query": query}
        if time:
            params["time"] = time.timestamp()
        
        try:
            url = f"{self.base_url}/api/v1/query"
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            
            result_data = response.json()
            logger.info("Prometheus instant query executed", 
                       query=query, status=result_data.get("status"))
            
            return PrometheusQueryResult(**result_data)
            
        except httpx.HTTPStatusError as e:
            logger.error("Prometheus query HTTP error", 
                        query=query, status_code=e.response.status_code,
                        response_text=e.response.text)
            raise
        except Exception as e:
            logger.error("Prometheus query failed", 
                        query=query, error=str(e))
            raise
    
    async def query_range(
        self, 
        query: str, 
        start: datetime, 
        end: datetime, 
        step: str = "1m"
    ) -> PrometheusQueryResult:
        """
        範囲クエリ実行
        
        Args:
            query: PromQLクエリ
            start: 開始時刻
            end: 終了時刻  
            step: ステップ間隔（例: "1m", "5m", "1h"）
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }
        
        try:
            url = f"{self.base_url}/api/v1/query_range"
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            
            result_data = response.json()
            logger.info("Prometheus range query executed", 
                       query=query, start=start.isoformat(), 
                       end=end.isoformat(), step=step,
                       status=result_data.get("status"))
            
            return PrometheusQueryResult(**result_data)
            
        except httpx.HTTPStatusError as e:
            logger.error("Prometheus range query HTTP error", 
                        query=query, status_code=e.response.status_code,
                        response_text=e.response.text)
            raise
        except Exception as e:
            logger.error("Prometheus range query failed", 
                        query=query, error=str(e))
            raise
    
    async def get_label_values(self, label: str) -> List[str]:
        """ラベル値一覧取得"""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        try:
            url = f"{self.base_url}/api/v1/label/{label}/values"
            response = await self._client.get(url)
            response.raise_for_status()
            
            result_data = response.json()
            if result_data.get("status") == "success":
                values = result_data.get("data", [])
                logger.info("Label values retrieved", 
                           label=label, values_count=len(values))
                return values
            else:
                logger.warning("Failed to get label values", 
                              label=label, status=result_data.get("status"))
                return []
                
        except Exception as e:
            logger.error("Get label values failed", 
                        label=label, error=str(e))
            return []
    
    async def get_series(self, match: List[str], start: Optional[datetime] = None, 
                        end: Optional[datetime] = None) -> List[Dict[str, str]]:
        """シリーズメタデータ取得"""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        params = {}
        for m in match:
            params.setdefault("match[]", []).append(m)
        
        if start:
            params["start"] = start.timestamp()
        if end:
            params["end"] = end.timestamp()
        
        try:
            url = f"{self.base_url}/api/v1/series"
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            
            result_data = response.json()
            if result_data.get("status") == "success":
                series = result_data.get("data", [])
                logger.info("Series metadata retrieved", 
                           match=match, series_count=len(series))
                return series
            else:
                logger.warning("Failed to get series", 
                              match=match, status=result_data.get("status"))
                return []
                
        except Exception as e:
            logger.error("Get series failed", 
                        match=match, error=str(e))
            return []
    
    def parse_instant_result(self, result: PrometheusQueryResult) -> List[PrometheusMetric]:
        """即座クエリ結果をパース"""
        metrics = []
        
        if result.status != "success" or not result.data:
            return metrics
        
        result_type = result.data.get("resultType")
        result_data = result.data.get("result", [])
        
        if result_type == "vector":
            for item in result_data:
                metric = item.get("metric", {})
                value_data = item.get("value", [])
                
                if len(value_data) == 2:
                    timestamp = datetime.fromtimestamp(float(value_data[0]))
                    value = float(value_data[1])
                    
                    # メトリクス名を取得（__name__ラベルから）
                    metric_name = metric.pop("__name__", "unknown")
                    
                    metrics.append(PrometheusMetric(
                        metric_name=metric_name,
                        labels=metric,
                        timestamp=timestamp,
                        value=value
                    ))
        
        logger.info("Instant result parsed", 
                   result_type=result_type, metrics_count=len(metrics))
        return metrics
    
    def parse_range_result(self, result: PrometheusQueryResult) -> List[MetricRange]:
        """範囲クエリ結果をパース"""
        ranges = []
        
        if result.status != "success" or not result.data:
            return ranges
        
        result_type = result.data.get("resultType")
        result_data = result.data.get("result", [])
        
        if result_type == "matrix":
            for item in result_data:
                metric = item.get("metric", {})
                values_data = item.get("values", [])
                
                # メトリクス名を取得
                metric_name = metric.pop("__name__", "unknown")
                
                # 値の配列を変換
                values = []
                for value_pair in values_data:
                    if len(value_pair) == 2:
                        timestamp = datetime.fromtimestamp(float(value_pair[0]))
                        value = float(value_pair[1])
                        values.append((timestamp, value))
                
                ranges.append(MetricRange(
                    metric_name=metric_name,
                    labels=metric,
                    values=values
                ))
        
        logger.info("Range result parsed", 
                   result_type=result_type, ranges_count=len(ranges))
        return ranges