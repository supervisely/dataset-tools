import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


HEATMAP_STATUS_CUSTOM_DATA_KEY = "heatmap_status"
HEATMAP_STATUS_SCHEMA_VERSION = 1
RUNNING_HEATMAP_STATUSES = {"queued", "running"}
TERMINAL_HEATMAP_STATUSES = {"success", "skipped", "failed", "stale"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def normalize_heatmap_status(
    status: Optional[Dict[str, Any]],
    stale_after_seconds: int = 10 * 60,
) -> Dict[str, Any]:
    if status is None:
        return {
            "schema_version": HEATMAP_STATUS_SCHEMA_VERSION,
            "status": "unknown",
            "stage": None,
            "message": "Heatmap generation status is not available.",
            "updated_at": None,
        }

    result = dict(status)
    if result.get("status") not in RUNNING_HEATMAP_STATUSES:
        return result

    updated_at = _parse_iso(result.get("updated_at"))
    if updated_at is None:
        result["status"] = "stale"
        result["message"] = (
            "Heatmap generation status has no update timestamp. The job may have been interrupted."
        )
        result["stale_after_seconds"] = stale_after_seconds
        return result

    age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    if age_seconds > stale_after_seconds:
        result["status"] = "stale"
        result["message"] = (
            "Heatmap generation status was not updated before the timeout. "
            "The job may have been interrupted."
        )
        result["stale_after_seconds"] = stale_after_seconds

    return result


def get_heatmap_status(
    api,
    project_id: int,
    stale_after_seconds: int = 10 * 60,
) -> Dict[str, Any]:
    project_info = api.project.get_info_by_id(project_id)
    custom_data = project_info.custom_data or {}
    status = custom_data.get(HEATMAP_STATUS_CUSTOM_DATA_KEY)
    return normalize_heatmap_status(status, stale_after_seconds)


def heatmap_status_endpoint(
    api,
    project_id: int,
    stale_after_seconds: int = 10 * 60,
) -> Dict[str, Any]:
    return get_heatmap_status(api, project_id, stale_after_seconds)


class HeatmapStatusReporter:
    def __init__(
        self,
        api,
        project_id: int,
        logger=None,
    ) -> None:
        self.api = api
        self.project_id = project_id
        self.logger = logger
        self.started_at = _utcnow_iso()

    def running(
        self,
        stage: str,
        message: str,
        progress: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.write(
            status="running",
            stage=stage,
            message=message,
            progress=progress,
            output_path=output_path,
        )

    def success(
        self,
        message: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.write(
            status="success",
            stage="done",
            message=message,
            progress=1,
            output_path=output_path,
        )

    def skipped(
        self,
        message: str,
        stage: str = "skipped",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.write(
            status="skipped",
            stage=stage,
            message=message,
            output_path=output_path,
        )

    def failed(
        self,
        error: BaseException,
        stage: str,
        message: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.write(
            status="failed",
            stage=stage,
            message=message or "Heatmap generation failed.",
            output_path=output_path,
            error={
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            },
        )

    def write(
        self,
        status: str,
        stage: Optional[str],
        message: str,
        progress: Optional[float] = None,
        output_path: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utcnow_iso()
        payload = {
            "schema_version": HEATMAP_STATUS_SCHEMA_VERSION,
            "status": status,
            "stage": stage,
            "message": message,
            "project_id": self.project_id,
            "progress": progress,
            "started_at": self.started_at,
            "updated_at": now,
            "finished_at": now if status in TERMINAL_HEATMAP_STATUSES else None,
            "output_path": output_path,
            "error": error,
        }
        self._update_project_custom_data(payload)
        return payload

    def _update_project_custom_data(self, payload: Dict[str, Any]) -> None:
        try:
            project_info = self.api.project.get_info_by_id(self.project_id)
            custom_data = dict(project_info.custom_data or {})
            custom_data[HEATMAP_STATUS_CUSTOM_DATA_KEY] = payload
            self.api.project.update_custom_data(
                self.project_id, custom_data, silent=True
            )
        except Exception as exc:
            self._warn(f"Failed to update heatmap status in project custom data: {exc}")

    def _warn(self, message: str) -> None:
        if self.logger is None:
            return
        warn = getattr(self.logger, "warning", None) or getattr(self.logger, "warn", None)
        if warn is not None:
            warn(message)
