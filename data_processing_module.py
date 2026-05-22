import csv
import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    import pandas as pd
except Exception:
    pd = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "upload"
MAX_PREVIEW_ROWS = 100
MAX_POINT_ROWS = 5000
MAX_MESH_TRIANGLES = 8000
MAX_SAMPLE_POINTS = 200000
MAX_RETURN_POINTS = 20000

router = APIRouter(prefix="/data-processing", tags=["data-processing"])


class GeometrySampleRequest(BaseModel):
    taskid: str = "default"
    dataset_id: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None
    sample_points: int = Field(default=10000, ge=1, le=MAX_SAMPLE_POINTS)
    return_points: int = Field(default=5000, ge=1, le=MAX_RETURN_POINTS)
    include_boundary: bool = True
    include_interior: bool = True
    compute_sdf_derivatives: bool = True
    device: Optional[str] = None
    seed: Optional[int] = None


def _safe_filename(filename: str) -> str:
    name = Path(filename or "uploaded-file").name
    return name.replace("\\", "_").replace("/", "_")


def _dataset_dir(taskid: str) -> Path:
    safe_taskid = "".join(ch for ch in taskid if ch.isalnum() or ch in "-_")[:80] or "default"
    path = UPLOAD_ROOT / safe_taskid / "data_processing"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _default_numeric_columns(count: int) -> List[str]:
    if count == 3:
        return ["x", "y", "z"]
    if count == 4:
        return ["x", "y", "z", "value"]
    if count == 5:
        return ["x", "y", "z", "field_1", "temperature"]
    return [f"col_{i + 1}" for i in range(count)]


def _detect_type(filename: str, columns: Optional[List[str]] = None) -> str:
    ext = Path(filename).suffix.lower()
    lower_columns = {c.strip().lower() for c in columns or []}
    if ext in {".csv", ".txt", ".dat"}:
        if {"x", "y", "z"}.issubset(lower_columns):
            return "point_cloud"
        return "table"
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext == ".stl":
        return "geometry_stl"
    if ext in {".obj", ".ply"}:
        return "geometry"
    return "unknown"


def _csv_read_options(path: Path) -> Dict[str, Any]:
    first_line = ""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                first_line = line.strip()
                break
    if not first_line:
        return {}
    if "," in first_line:
        tokens = [part.strip() for part in first_line.split(",")]
        sep = ","
    else:
        tokens = first_line.split()
        sep = r"\s+"
    options: Dict[str, Any] = {"sep": sep}
    if sep != ",":
        options["engine"] = "python"
    if tokens and all(_is_number(token) for token in tokens):
        options["header"] = None
        options["names"] = _default_numeric_columns(len(tokens))
    return options


def _read_csv_preview(path: Path) -> Dict[str, Any]:
    if pd is not None:
        df = pd.read_csv(path, nrows=max(MAX_POINT_ROWS, MAX_PREVIEW_ROWS), **_csv_read_options(path))
        columns = [str(c) for c in df.columns]
        rows = df.head(MAX_PREVIEW_ROWS).replace({np.nan: None}).to_dict(orient="records")
        result: Dict[str, Any] = {"columns": columns, "rows": rows, "row_preview_count": len(rows)}
        lower_map = {c.lower(): c for c in columns}
        if {"x", "y", "z"}.issubset(lower_map):
            point_df = df[[lower_map["x"], lower_map["y"], lower_map["z"]]].copy()
            point_df.columns = ["x", "y", "z"]
            for optional_key in ["sdf", "value", "temperature"]:
                if optional_key in lower_map:
                    point_df[optional_key if optional_key != "temperature" else "value"] = df[lower_map[optional_key]]
            point_df = point_df.apply(pd.to_numeric, errors="coerce").dropna().head(MAX_POINT_ROWS)
            result["points"] = point_df.to_dict(orient="records")
        return result

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.readline()
        f.seek(0)
        if sample and "," not in sample:
            rows, points = [], []
            first_tokens = sample.strip().split()
            has_header = not all(_is_number(token) for token in first_tokens)
            columns = first_tokens if has_header else _default_numeric_columns(len(first_tokens))
            lower_map = {c.lower(): c for c in columns}
            for idx, line in enumerate(f):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if idx == 0 and has_header:
                    continue
                row = {columns[i]: tokens[i] for i in range(min(len(columns), len(tokens)))}
                if len(rows) < MAX_PREVIEW_ROWS:
                    rows.append(row)
                if {"x", "y", "z"}.issubset(lower_map) and len(points) < MAX_POINT_ROWS:
                    try:
                        point = {
                            "x": float(row[lower_map["x"]]),
                            "y": float(row[lower_map["y"]]),
                            "z": float(row[lower_map["z"]]),
                        }
                        if "value" in lower_map:
                            point["value"] = float(row[lower_map["value"]])
                        elif "temperature" in lower_map:
                            point["value"] = float(row[lower_map["temperature"]])
                        points.append(point)
                    except Exception:
                        pass
            result = {"columns": columns, "rows": rows, "row_preview_count": len(rows)}
            if points:
                result["points"] = points
            return result

        reader = csv.DictReader(f)
        rows, points = [], []
        columns = reader.fieldnames or []
        lower_map = {c.lower(): c for c in columns}
        for idx, row in enumerate(reader):
            if idx < MAX_PREVIEW_ROWS:
                rows.append(row)
            if {"x", "y", "z"}.issubset(lower_map) and len(points) < MAX_POINT_ROWS:
                try:
                    points.append({
                        "x": float(row[lower_map["x"]]),
                        "y": float(row[lower_map["y"]]),
                        "z": float(row[lower_map["z"]]),
                    })
                except Exception:
                    pass
            if idx >= MAX_POINT_ROWS:
                break
        result = {"columns": columns, "rows": rows, "row_preview_count": len(rows)}
        if points:
            result["points"] = points
        return result


def _read_json_preview(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines()[:MAX_PREVIEW_ROWS] if line.strip()]
        return {"rows": rows, "row_preview_count": len(rows)}
    data = json.loads(text)
    if isinstance(data, list):
        return {"rows": data[:MAX_PREVIEW_ROWS], "row_preview_count": min(len(data), MAX_PREVIEW_ROWS)}
    if isinstance(data, dict):
        return {"object": data}
    return {"value": data}


def _read_stl_preview(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 84:
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = ""
    triangles: List[List[List[float]]] = []
    triangle_count = 0
    if len(raw) >= 84:
        binary_count = struct.unpack("<I", raw[80:84])[0]
        expected_size = 84 + binary_count * 50
        if expected_size == len(raw):
            triangle_count = int(binary_count)
            offset = 84
            for _ in range(min(triangle_count, MAX_MESH_TRIANGLES)):
                values = struct.unpack("<12fH", raw[offset:offset + 50])
                triangles.append([
                    [values[3], values[4], values[5]],
                    [values[6], values[7], values[8]],
                    [values[9], values[10], values[11]],
                ])
                offset += 50
    if not triangles:
        text = text or raw.decode("utf-8", errors="ignore")
        current: List[List[float]] = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                try:
                    current.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    current = []
            if len(current) == 3:
                triangle_count += 1
                if len(triangles) < MAX_MESH_TRIANGLES:
                    triangles.append(current)
                current = []
    if not triangles:
        raise HTTPException(status_code=400, detail="No STL triangles found")
    points = np.array([p for tri in triangles for p in tri], dtype=float)
    return {
        "triangle_count": triangle_count or len(triangles),
        "triangles_preview_count": len(triangles),
        "triangles": triangles,
        "bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
    }


def _build_preview(path: Path, filename: str) -> Dict[str, Any]:
    ext = path.suffix.lower()
    columns: Optional[List[str]] = None
    if ext in {".csv", ".txt", ".dat"}:
        preview = _read_csv_preview(path)
        columns = preview.get("columns")
    elif ext in {".json", ".jsonl"}:
        preview = _read_json_preview(path)
    elif ext == ".stl":
        preview = _read_stl_preview(path)
    else:
        preview = {"message": "Preview is not available for this file type yet."}
    return {"data_type": _detect_type(filename, columns), "preview": preview}


def _array_to_points(data: Dict[str, Any], limit: int) -> List[Dict[str, float]]:
    if not {"x", "y", "z"}.issubset(data):
        return []
    keys = [k for k in [
        "x", "y", "z", "normal_x", "normal_y", "normal_z",
        "sdf", "sdf__x", "sdf__y", "sdf__z", "area",
    ] if k in data]
    arrays = {k: np.asarray(data[k]).reshape(-1) for k in keys}
    count = min(limit, *(len(v) for v in arrays.values()))
    points = []
    for i in range(count):
        item = {}
        for key in keys:
            try:
                item[key] = float(arrays[key][i])
            except Exception:
                pass
        points.append(item)
    return points


def _find_dataset_file(taskid: str, dataset_id: Optional[str], filename: Optional[str]) -> Path:
    candidates = [p for p in _dataset_dir(taskid).iterdir() if p.is_file()]
    if dataset_id:
        candidates = [p for p in candidates if p.name.startswith(dataset_id)]
    if filename:
        safe = _safe_filename(filename)
        candidates = [p for p in candidates if p.name.endswith(safe) or p.name == safe]
    stl_candidates = [p for p in candidates if p.suffix.lower() == ".stl"]
    if stl_candidates:
        return sorted(stl_candidates)[-1]
    if candidates:
        return sorted(candidates)[-1]
    raise HTTPException(status_code=404, detail="Dataset file not found")


def _resolve_geometry_path(req: GeometrySampleRequest) -> Path:
    if req.path:
        path = Path(req.path).resolve()
        base_dir = BASE_DIR.resolve()
        if not str(path).startswith(str(base_dir)):
            raise HTTPException(status_code=400, detail="Path must be under the service workspace")
        if not path.exists():
            raise HTTPException(status_code=404, detail="File path not found")
        return path
    return _find_dataset_file(req.taskid, req.dataset_id, req.filename)


def _geometry_engine_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    try:
        import physicsnemo  # type: ignore
        import physicsnemo.sym  # type: ignore
        status["physicsnemo_available"] = True
        status["physicsnemo_path"] = getattr(physicsnemo, "__file__", None)
    except Exception as exc:
        status["physicsnemo_available"] = False
        status["physicsnemo_error"] = str(exc)
    try:
        from src.geometry import Tessellation  # noqa: F401
        status["local_tessellation_available"] = True
    except Exception as exc:
        status["local_tessellation_available"] = False
        status["local_tessellation_error"] = str(exc)
    return status


def _load_tessellation(path: Path, req: GeometrySampleRequest):
    try:
        from physicsnemo.sym.geometry.tessellation_warp import Tessellation
        try:
            return "physicsnemo.sym.geometry.tessellation_warp.Tessellation", Tessellation.from_stl(
                str(path), airtight=True, device=req.device, seed=req.seed
            )
        except TypeError:
            return "physicsnemo.sym.geometry.tessellation_warp.Tessellation", Tessellation.from_stl(str(path), airtight=True)
    except Exception:
        from src.geometry import Tessellation
        return "src.geometry.tessellation.Tessellation", Tessellation.from_stl(str(path))


def _sample_geometry(req: GeometrySampleRequest) -> Dict[str, Any]:
    stl_path = _resolve_geometry_path(req)
    if stl_path.suffix.lower() != ".stl":
        raise HTTPException(status_code=400, detail="Geometry sampling currently requires an STL file")
    engine, geometry = _load_tessellation(stl_path, req)
    result: Dict[str, Any] = {
        "engine": engine,
        "source": str(stl_path),
        "sample_points": req.sample_points,
        "return_points": req.return_points,
    }
    if req.include_boundary:
        boundary = geometry.sample_boundary(nr_points=req.sample_points)
        result["boundary"] = {
            "points": _array_to_points(boundary, req.return_points),
            "keys": sorted(boundary.keys()),
            "total_points": int(req.sample_points),
        }
        if "area" in boundary:
            result["boundary"]["area_sum"] = float(np.asarray(boundary["area"]).sum())
    if req.include_interior:
        interior = geometry.sample_interior(
            nr_points=req.sample_points,
            compute_sdf_derivatives=req.compute_sdf_derivatives,
        )
        result["interior"] = {
            "points": _array_to_points(interior, req.return_points),
            "keys": sorted(interior.keys()),
            "total_points": int(req.sample_points),
        }
        if "area" in interior:
            result["interior"]["area_sum"] = float(np.asarray(interior["area"]).sum())
    return result


@router.get("/health")
def data_processing_health() -> Dict[str, Any]:
    return {"status": "ok", "module": "data-processing"}


@router.get("/physicsnemo/status")
def physicsnemo_status() -> Dict[str, Any]:
    return _geometry_engine_status()


@router.post("/upload-preview")
async def upload_preview(file: UploadFile = File(...), taskid: str = Form("default")) -> JSONResponse:
    filename = _safe_filename(file.filename or "uploaded-file")
    dataset_id = f"ds_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    save_path = _dataset_dir(taskid) / f"{dataset_id}_{filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    save_path.write_bytes(content)
    result = _build_preview(save_path, filename)
    return JSONResponse({
        "dataset_id": dataset_id,
        "filename": filename,
        "size": len(content),
        "path": str(save_path),
        **result,
    })


@router.post("/geometry/sample")
def geometry_sample(req: GeometrySampleRequest) -> Dict[str, Any]:
    return _sample_geometry(req)


@router.get("/files")
def data_processing_files(taskid: str = "default") -> Dict[str, Any]:
    files = [p.name for p in sorted(_dataset_dir(taskid).iterdir()) if p.is_file()]
    return {"taskid": taskid, "files": files}


@router.get("/demo", response_class=HTMLResponse)
def data_processing_demo() -> str:
    return HTML_DEMO


def create_app() -> FastAPI:
    app = FastAPI(title="Science42 Data Processing")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()


HTML_DEMO = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Science42 数据处理预览</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #eef4fb; color: #172033; }
    .shell { max-width: 1180px; margin: 32px auto; padding: 0 20px; }
    .panel { background: rgba(255,255,255,.92); border: 1px solid #d8e4f2; border-radius: 8px; box-shadow: 0 16px 48px rgba(34,74,124,.12); overflow: hidden; }
    .head { padding: 18px 22px; border-bottom: 1px solid #d8e4f2; display:flex; justify-content:space-between; gap:16px; align-items:center; }
    h1 { font-size: 20px; margin: 0; }
    .upload { padding: 18px 22px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    input[type=file] { border: 1px dashed #7fa5cf; padding: 12px; border-radius: 6px; background: #f8fbff; }
    button { border: 0; background: #1f63b7; color: white; padding: 11px 16px; border-radius: 6px; cursor: pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .meta { padding: 0 22px 18px; color: #526070; font-size: 13px; }
    .grid { display:grid; grid-template-columns: 1.1fr .9fr; border-top:1px solid #d8e4f2; }
    .view, .json { padding: 18px 22px; min-height: 420px; }
    .view { border-right: 1px solid #d8e4f2; background:#fff; }
    pre { white-space: pre-wrap; word-break: break-word; background:#0e1726; color:#dbeafe; padding:14px; border-radius:6px; overflow:auto; max-height:520px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4edf7; padding: 8px; text-align: left; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    th { background: #f5f9fe; position: sticky; top: 0; }
    canvas { width: 100%; height: 420px; background: linear-gradient(180deg, #f8fbff, #eaf2fb); border: 1px solid #d8e4f2; border-radius: 6px; }
    .hint { color:#607086; margin-top:12px; font-size:13px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .view { border-right:0; border-bottom:1px solid #d8e4f2; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="head"><h1>Science42 数据处理预览</h1><span id="status">等待上传</span></div>
      <div class="upload">
        <input id="file" type="file" accept=".csv,.txt,.dat,.json,.jsonl,.stl" />
        <button id="submit">上传并预览</button>
      </div>
      <div class="meta" id="meta"></div>
      <div class="grid"><div class="view" id="view"><div class="hint">CSV/TXT 会展示表格；包含 x,y,z 的数据会展示点云；STL 会展示几何投影。</div></div><div class="json"><pre id="json">{}</pre></div></div>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById('file'), submit = document.getElementById('submit'), statusEl = document.getElementById('status'), metaEl = document.getElementById('meta'), viewEl = document.getElementById('view'), jsonEl = document.getElementById('json');
    submit.onclick = async () => {
      const file = fileInput.files[0]; if (!file) return; submit.disabled = true; statusEl.textContent = '处理中...';
      const fd = new FormData(); fd.append('file', file); fd.append('taskid', 'demo');
      try {
        const res = await fetch('/data-processing/upload-preview', { method: 'POST', body: fd });
        const data = await res.json(); if (!res.ok) throw new Error(data.detail || data.error || '请求失败');
        jsonEl.textContent = JSON.stringify(data, null, 2);
        metaEl.textContent = `${data.filename} | ${data.data_type} | ${data.size} bytes | ${data.dataset_id}`;
        render(data); statusEl.textContent = '预览完成';
      } catch (err) { statusEl.textContent = '处理失败'; viewEl.innerHTML = `<div class="hint">${escapeHtml(err.message)}</div>`; }
      finally { submit.disabled = false; }
    };
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function render(data) {
      const preview = data.preview || {};
      if (preview.points && preview.points.length) return renderCanvas(preview.points, []);
      if (preview.triangles && preview.triangles.length) return renderCanvas([], preview.triangles);
      if (preview.rows && preview.rows.length) return renderTable(preview.rows);
      viewEl.innerHTML = `<div class="hint">${escapeHtml(preview.message || '暂无可视化预览')}</div>`;
    }
    function renderTable(rows) {
      const columns = Object.keys(rows[0] || {});
      viewEl.innerHTML = `<div style="overflow:auto; max-height:520px"><table><thead><tr>${columns.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${escapeHtml(r[c] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
    function renderCanvas(points, triangles) {
      viewEl.innerHTML = '<canvas id="canvas" width="900" height="520"></canvas>';
      const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
      const all = points.length ? points : triangles.flat();
      const xs = all.map(p => p.x ?? p[0]), ys = all.map(p => p.y ?? p[1]), zs = all.map(p => p.z ?? p[2]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys), minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const scale = Math.min(canvas.width / Math.max(maxX - minX, 1), canvas.height / Math.max(maxY - minY, 1)) * 0.75;
      const project = p => {
        const x = p.x ?? p[0], y = p.y ?? p[1], z = p.z ?? p[2];
        return [canvas.width / 2 + (x - (minX + maxX)/2) * scale + (z - (minZ + maxZ)/2) * scale * 0.22, canvas.height / 2 - (y - (minY + maxY)/2) * scale + (z - (minZ + maxZ)/2) * scale * 0.12];
      };
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if (triangles.length) {
        ctx.strokeStyle = '#1f63b7'; ctx.globalAlpha = 0.22;
        triangles.forEach(t => { const a = project(t[0]), b = project(t[1]), c = project(t[2]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath(); ctx.stroke(); });
      } else {
        ctx.fillStyle = '#1f63b7'; points.forEach(p => { const q = project(p); ctx.fillRect(q[0], q[1], 2, 2); });
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("data_processing_module:app", host="0.0.0.0", port=1212, reload=False)
