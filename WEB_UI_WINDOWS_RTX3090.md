# Web UI 1 ảnh / 4 ảnh trên Windows + RTX 3090

Tài liệu này dành cho bản shape-only đã được kiểm chứng với:

- Windows 11, Python 3.11.9.
- NVIDIA GeForce RTX 3090, CUDA 12.4.
- `tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv`.
- Weight `model.fp16.safetensors` trong cache của workspace.
- Gradio 5.33.0, FastAPI 0.115.12 và Uvicorn 0.34.3.

## Trạng thái đã kiểm chứng

Web UI đã được khởi động thật tại `http://127.0.0.1:8080` và endpoint `/health` trả về trạng thái `ready`, `multiview=true`, `device=cuda`.

Hai chế độ đã được gửi qua chính endpoint Gradio `/shape_generation` với 30 inference steps, guidance 5.0 và octree resolution 256:

- Tab `1 ẢNH · Single View`: dùng view `front`, tổng thời gian 27,24 giây, output 132.274 vertices và 264.540 faces.
- Tab `4 ẢNH · Multi View`: dùng `front`, `left`, `back`, `right`, tổng thời gian 46,30 giây, output 124.078 vertices và 248.160 faces.
- GLB đã được nạp lại bằng Trimesh; vertex hữu hạn và face index hợp lệ.

## Mở Web UI bằng một lệnh

Từ CMD hoặc PowerShell tại thư mục gốc của source, chạy nguyên một dòng:

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\start_windows_multiview_webui.ps1" -Background -OpenBrowser
```

Launcher sẽ:

1. Kiểm tra dependency, CUDA, RTX 3090 và model fp16 trong cache.
2. Mở server ngầm chỉ trên máy cục bộ.
3. Chờ model nạp xong rồi mở `http://127.0.0.1:8080`.
4. Ghi PID và log để có thể kiểm tra hoặc dừng server.

Nếu trình duyệt không tự mở, truy cập thủ công:

```text
http://127.0.0.1:8080
```

Chạy lại lệnh start khi server đang hoạt động sẽ không tạo tiến trình trùng; launcher chỉ thông báo URL hiện tại.

## Dừng Web UI

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\start_windows_multiview_webui.ps1" -Stop
```

Lệnh này sử dụng PID server thật do endpoint health cung cấp và đã được kiểm thử trên Windows.

## Chỉ kiểm tra môi trường

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\start_windows_multiview_webui.ps1" -PreflightOnly
```

## Cách sử dụng giao diện

1. Chọn tab `1 ẢNH · Single View` nếu chỉ có một ảnh chính diện; tải ảnh vào `Ảnh chính diện · Front`.
2. Chọn tab `4 ẢNH · Multi View` nếu có đủ bốn hướng; tải đúng Front, Back, Left và Right vào bốn ô được đánh số.
3. Giữ thiết lập đầu tiên: Steps 30, Guidance 5.0, Octree Resolution 256, Number of Chunks 8000.
4. Tắt `Randomize seed` khi cần tái tạo đúng cùng một kết quả.
5. Nhấn `Generate 3D · 1 Image` hoặc `Generate 3D · 4 Images` và chờ hàng đợi hoàn tất.
6. Xem mesh ở khung `Generated Mesh`.
7. Tải GLB từ trường `Generated mesh (direct download)`.

Tab một ảnh yêu cầu đúng một ảnh chính diện. Tab bốn ảnh yêu cầu đủ cả bốn hướng; backend chỉ đọc dữ liệu của tab đang được chọn.

## Yêu cầu ảnh đầu vào

- Dùng PNG RGBA có nền trong suốt.
- Với tab bốn ảnh, tất cả ảnh phải là cùng một vật thể, không đổi tư thế hoặc hình dạng.
- Vật thể nằm giữa ảnh và có tỷ lệ gần giống nhau.
- Camera có độ cao, khoảng cách và tiêu cự gần giống nhau.
- Left/Right là camera quay quanh vật thể; không dùng ảnh Front lật gương.

Dependency tối thiểu không cài `rembg`, vì bản này ưu tiên PNG đã tách nền và tránh tải U2Net khi mở web. Control Remove Background sẽ tự ẩn khi package không có.

## Output và log

Mỗi lượt tạo mesh nằm trong một thư mục UUID tại:

```text
hy3dshape\output_folder\webui\<uuid>\white_mesh.glb
```

Log server và PID nằm tại:

```text
hy3dshape\output_folder\webui\logs
```

## Giới hạn hiện tại

- UI đang chạy shape-only; launcher luôn truyền `--disable_tex`.
- GLB hiện là geometry chưa có texture/PBR.
- Export nâng cao và Simplify dùng PyMeshLab sẽ tự ẩn khi dependency tùy chọn chưa được cài.
- Server mặc định chỉ bind `127.0.0.1`, vì UI chưa có đăng nhập. Không mở ra Internet công cộng.
- Xem quy trình CLI tại `MULTIVIEW_WINDOWS_RTX3090.md` nếu không muốn dùng trình duyệt.
