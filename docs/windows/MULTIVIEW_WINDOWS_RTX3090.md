# Hunyuan3D 4-view to 3D trên Windows + RTX 3090

Tài liệu này ghi lại cấu hình multi-view đã chạy thành công trên máy hiện tại ngày 2026-07-17. Luồng nhận bốn ảnh của cùng một vật thể theo hướng trước, trái, sau và phải rồi sinh ra **một mesh GLB**.

## Trạng thái đã xác nhận

- Model: `tencent/Hunyuan3D-2mv`.
- Subfolder: `hunyuan3d-dit-v2-mv`.
- Checkpoint: `model.fp16.safetensors`, 4.928.151.562 byte.
- GPU: NVIDIA GeForce RTX 3090 24 GB.
- Tensor đầu vào sau preprocess: `(1, 4, 3, 512, 512)`; đây là một object có bốn view, không phải bốn object riêng.
- Latent smoke test: `(1, 3072, 64)`, FP16 CUDA.
- Inference đầy đủ: 30 step, octree resolution 256, guidance 5, seed 12345.
- Kết quả: 125.686 vertices, 251.360 faces, mesh watertight, thời gian 60,2 giây.

Kết quả thử đã được lưu tại:

```text
hy3dshape/output_folder/inference/multiview_sample_4views.glb
```

## Source đã được điều chỉnh

- `hy3dshape/hy3dshape/pipelines.py` ánh xạ namespace checkpoint 2.0 cũ `hy3dgen.shapegen.*` sang package `hy3dshape.*` của source 2.1.
- `hy3dshape/hy3dshape/utils/utils.py` chỉ tải `config.yaml` và đúng định dạng trọng số được yêu cầu. Với 2mv safetensors, cách này tải 4,93 GB thay vì tải cả safetensors và checkpoint trùng nhau tổng cộng khoảng 9,86 GB.
- `hy3dshape/scripts/infer_windows_multiview.py` kiểm tra bốn PNG RGBA, load model và xuất GLB.
- `hy3dshape/scripts/infer_windows_multiview.ps1` thiết lập CUDA/cache và cung cấp lệnh một dòng dùng được từ CMD.

Model được lưu trong cache của repository:

```text
.cache/hy3dgen/tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/
├── config.yaml
└── model.fp16.safetensors
```

## Chạy lại bộ mẫu

Tại thư mục gốc repository, chạy một dòng trong CMD hoặc PowerShell:

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\infer_windows_multiview.ps1"
```

Launcher mặc định dùng bốn render của cùng một object trong mini dataset:

| Hướng | File mẫu |
|---|---|
| Front | `007.png` |
| Left | `005.png` |
| Back | `006.png` |
| Right | `004.png` |

Bốn camera mẫu cách nhau đúng 90 độ và có FOV/khoảng cách gần tương đương. Chúng phù hợp cho smoke test kỹ thuật; dữ liệu thực nên dùng camera đồng bộ tốt hơn.

## Chạy bằng bốn ảnh của bạn

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\infer_windows_multiview.ps1" -Front "E:\anh_4_huong\front.png" -Left "E:\anh_4_huong\left.png" -Back "E:\anh_4_huong\back.png" -Right "E:\anh_4_huong\right.png" -Output ".\hy3dshape\output_folder\inference\vat_the_cua_toi.glb"
```

Yêu cầu cho cả bốn ảnh:

- PNG RGBA có nền trong suốt.
- Cùng một vật thể và không thay đổi tư thế/hình dạng.
- Vật thể nằm giữa ảnh, cùng tỷ lệ tương đối.
- Camera gần cùng độ cao, khoảng cách và tiêu cự.
- `left` và `right` là góc nhìn khi camera quay quanh vật thể, không phải lật gương ảnh front.

## Chất lượng cao hơn

Sau khi test 256 thành công, có thể tăng giải mã mesh:

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\hy3dshape\scripts\infer_windows_multiview.ps1" -Front "E:\anh_4_huong\front.png" -Left "E:\anh_4_huong\left.png" -Back "E:\anh_4_huong\back.png" -Right "E:\anh_4_huong\right.png" -Output ".\hy3dshape\output_folder\inference\vat_the_hq.glb" -Steps 30 -OctreeResolution 380
```

Resolution cao hơn làm bước VAE/mesh chậm hơn và dùng nhiều bộ nhớ hơn. Nên giữ 256 cho lần đầu của mỗi bộ ảnh.

## Giới hạn

- Đây là generative reconstruction, không phải photogrammetry có kích thước chính xác tuyệt đối.
- Output hiện là geometry thô, chưa chạy Hunyuan3D Paint/PBR.
- LoRA Hunyuan3D-2.1 single-view hiện tại không tương thích với kiến trúc Hunyuan3D-2mv 1.1B.
- Muốn train LoRA multi-view cần DataLoader lấy bốn ảnh đồng bộ cho mỗi object và config model/target module riêng; launcher training single-view hiện tại không thực hiện việc đó.
