# Huấn luyện Hunyuan3D 2.1 Shape LoRA trên Windows + RTX 3090

Tài liệu này mô tả cấu hình hiện tại của máy, phần source đã được điều chỉnh và quy trình chạy thử huấn luyện 2D → 3D trên Windows. Phạm vi hiện tại chỉ là **Shape DiT** (`hy3dshape`): ảnh 2D đầu vào được dùng để fine-tune khả năng sinh hình học 3D. `hy3dpaint` chưa nằm trong giai đoạn này.

> Trạng thái đã kiểm tra ngày 2026-07-16: môi trường Windows/CUDA, dataset mini, model pretrained và hai optimizer step LoRA thực tế đều đã chạy thành công trên RTX 3090. Adapter PEFT FP32 cuối đã được lưu, đọc lại và xác nhận có learned delta; có thể chuyển sang chuẩn bị dataset thật trước khi chạy pilot dài.

## 1. Cấu hình máy đã kiểm tra ngày 2026-07-16

| Thành phần | Cấu hình hiện tại | Ghi chú |
|---|---|---|
| Hệ điều hành | Windows 11 Enterprise LTSC, build 26100, 64-bit | Chạy native Windows trước, chưa cần Ubuntu |
| GPU | NVIDIA GeForce RTX 3090 | 24 GB VRAM, compute capability 8.6 |
| NVIDIA driver | 591.86 | Đủ mới để chạy PyTorch CUDA 12.4 |
| CUDA Toolkit cục bộ | 12.8, NVCC 12.8.61 | Không cần gỡ; wheel PyTorch cu124 mang runtime riêng |
| CPU | Intel Core i9-12900K | Windows hiện cung cấp 16 logical processors |
| RAM | 128 GB | Đủ cho DataLoader và model cache |
| Ổ E | Khoảng 152 GB trống tại lúc kiểm tra | Nên lưu source/output ở E |
| Ổ D | Khoảng 17 GB trống tại lúc kiểm tra | Không nên đặt model cache hoặc checkpoint ở D |
| Python mặc định | 3.14.5 | Không dùng cho project này |
| Python dùng cho project | 3.11.9 | `.venv-win` đã chạy thành công preflight và smoke test |
| WSL | Chỉ có distro nội bộ `docker-desktop` | Chưa cài Ubuntu người dùng |

## 2. Khả năng thực tế của RTX 3090

- Có thể chạy inference Shape pretrained.
- Có thể thử huấn luyện thật bằng LoRA trên denoiser pretrained, batch size 1, BF16 và activation checkpointing.
- Không thể chạy nguyên config full fine-tune chính thức; config đó ghi nhận huấn luyện trên 8 GPU H20 98 GB.
- Không thể chạy config mini nguyên bản; “mini” chỉ là dataset 8 object, còn model vẫn lớn và comment trong source ghi khoảng 68 GB VRAM.
- Không phù hợp để train model 3.3B từ đầu trên một RTX 3090.

LoRA vẫn là huấn luyện AI thật: gradient cập nhật chính xác 5.160.960 tham số adapter trong các projection attention của Shape DiT. ShapeVAE, DINO và trọng số nền của Shape DiT được giữ đóng băng.

## 3. Source đã được điều chỉnh

### LoRA và bộ nhớ

- `flow_matching_sit.py` hỗ trợ đầy đủ `rank`, `alpha`, `dropout`, `bias`, `target_modules` và `adapter_path`.
- Optimizer chỉ nhận tham số có `requires_grad=True`; nếu LoRA không gắn đúng module, chương trình dừng ngay thay vì train rỗng.
- `HunYuanDiTPlain` hỗ trợ checkpoint từng transformer block với `use_reentrant=False`, cần thiết khi model nền bị đóng băng bởi PEFT.
- Callback PEFT lưu riêng `adapter_model.safetensors` và `adapter_config.json`, không ghi lặp toàn bộ model nền nhiều GB.
- Trọng số nền giữ FP16 nhưng tham số LoRA trainable và trạng thái AdamW giữ FP32, tránh update nhỏ trong warm-up bị làm tròn về 0.
- Khi nạp adapter để train tiếp, pipeline giữ nguyên tensor PEFT FP32 thay vì cast FP16 rồi upcast lại, nên learned delta nhỏ không bị mất precision.
- Pipeline inference có `load_lora_adapter(...)`, mặc định merge adapter vào denoiser.
- Có chế độ `--smoke_test`: hai optimizer step, DataLoader worker bằng 0, tắt validation và tự kiểm tra LoRA phải thực sự thay đổi.

### Tương thích Windows và dữ liệu

- UID object được lấy bằng API đường dẫn đa nền tảng, không còn tách chuỗi bằng `/`.
- Nhánh đọc danh sách JSON đã sửa biến sai và xử lý đường dẫn tương đối theo vị trí file JSON.
- File NPZ được đóng đúng cách sau khi đọc.
- Loader dừng sau một số lỗi liên tiếp, tránh hiện tượng lặp vô hạn khi dữ liệu sai.
- Worker trên Windows được seed riêng cho Python, NumPy và RNG của dataset.
- Có validator cấu trúc dataset chạy được ngay cả trước khi cài NumPy/Torch.

### File mới

- Config: `hy3dshape/configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lora-rank8-rtx3090-windows.yaml`
- Launcher: `hy3dshape/scripts/train_windows_rtx3090_lora.ps1`
- Validator: `hy3dshape/tools/validate_shape_dataset.py`
- Environment preflight: `hy3dshape/tools/check_windows_training_env.py`
- Dependencies: `hy3dshape/requirements-windows-training.txt`

Các config và script Linux gốc vẫn được giữ lại để đối chiếu.

## 4. Cấu hình LoRA RTX 3090 hiện tại

| Tùy chọn | Giá trị | Ý nghĩa |
|---|---:|---|
| Base model | `tencent/Hunyuan3D-2.1` | Load ShapeVAE và Shape DiT pretrained |
| Batch size | 1 | Giảm VRAM đỉnh |
| Gradient accumulation | 8 | Effective batch tương đương 8 sample |
| Precision | BF16 | RTX 3090 hỗ trợ BF16 |
| Training steps | 10.000 | Giá trị pilot; cần điều chỉnh theo quy mô dataset thật |
| Learning rate | `5e-5` | Điểm bắt đầu thận trọng cho LoRA |
| LoRA rank/alpha | 8 / 16 | 5.160.960 tham số trainable (0,1689% denoiser) |
| Dtype LoRA / model nền | FP32 / FP16 | Giữ optimizer ổn định nhưng chỉ tăng ít VRAM vì model nền vẫn đóng băng |
| LoRA dropout | 0.05 | Regularization cho tập dữ liệu nhỏ |
| Target modules | `to_q`, `to_k`, `to_v`, `out_proj` | Self-attention và cross-attention |
| Latent tokens | 4096 | Giữ tương thích với model pretrained |
| Point samples | 81.920 | Giữ tương thích ShapeVAE pretrained |
| Image encoder | DINOv2 Large, 518 px | Giống config fine-tune gốc |
| Gradient checkpointing | Bật | Giảm activation VRAM, đổi lại train chậm hơn |
| Mesh callbacks | Tắt | Tránh spike VRAM trong lúc train |
| Full Lightning checkpoint | Tắt | Chỉ lưu adapter nhẹ |
| Lightning model summary | Tắt | Tránh tràn số `int32` khi Lightning 1.9.5 đếm model PEFT hơn 3 tỷ tham số trên Windows |
| Adapter interval | 500 optimizer steps | Thư mục `lora/step_XXXXXXXX` |

Không giảm 4096 latent tokens xuống 512 một cách tùy ý vì model pretrained/VAE hiện tại không được cấu hình cho thay đổi đó.

## 5. Dữ liệu bắt buộc

Chỉ một thư mục ảnh 2D là chưa đủ. Mỗi object huấn luyện phải có ground truth 3D đã được preprocess theo cấu trúc:

```text
preprocessed/
└── <uid>/
    ├── render_cond/
    │   ├── 000.png
    │   ├── ...
    │   └── 023.png
    └── geo_data/
        └── <uid>_surface.npz
```

Yêu cầu hiện tại:

- 24 ảnh PNG RGBA 8-bit cho mỗi object.
- NPZ có hai array `random_surface` và `sharp_surface`.
- Mỗi array có dạng `(N, 6)` hoặc nhiều hơn: XYZ + normal XYZ.
- `random_surface` phải có ít nhất 81.920 điểm vì sampling không hoàn lại.
- Chia train/validation theo UID object, không chia các góc nhìn của cùng một object sang hai tập.

Bộ mini đi kèm đã qua validator: 8 object, 192 ảnh RGBA và 8 surface archive đều đúng cấu trúc. Khi chưa có NumPy, validator bỏ qua kiểm tra NaN/infinity; khi chưa có Pillow, nó bỏ qua decode pixel/alpha. Kiểm tra cấu trúc PNG/CRC và header, shape, dtype của NPZ vẫn chạy bằng thư viện chuẩn.

## 6. Tạo môi trường Windows

Chạy tại thư mục gốc repository bằng PowerShell:

```powershell
py -3.11 -m venv .venv-win
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install setuptools==80.9.0
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r .\hy3dshape\requirements-windows-training.txt
```

File requirements cài đúng wheel `torch-cluster` cho Python 3.10/3.11, PyTorch 2.5 và CUDA 12.4 từ [kho wheel chính thức của PyTorch Geometric](https://data.pyg.org/whl/torch-2.5.0%2Bcu124.html). ShapeVAE cần package này cho farthest-point sampling ở batch đầu tiên.

Không dùng Python 3.14. Không cài `deepspeed`, NCCL, Blender, Paint dependencies hoặc toàn bộ requirements gốc trong môi trường thử Shape LoRA này.

Model pretrained Hunyuan3D và DINO sẽ được tải ở lần chạy đầu. Cần bảo đảm ổ chứa Hugging Face cache còn đủ dung lượng; tránh ổ D trong cấu hình máy hiện tại.

## 7. Kiểm tra dataset

Validator có thể chạy ngay bằng Python 3.11 hiện có:

```powershell
cd .\hy3dshape
py -3.11 .\tools\validate_shape_dataset.py `
  .\tools\mini_trainset\preprocessed `
  --views 24 --pc-size 81920 --pc-sharpedge-size 0
cd ..
```

Kết quả hợp lệ phải kết thúc bằng `Dataset validation PASSED.`

## 8. Smoke test bắt buộc

Sau khi cài môi trường, chạy hai optimizer step trước:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\hy3dshape\scripts\train_windows_rtx3090_lora.ps1 `
  -SmokeTest
```

Nếu chỉ muốn kiểm tra môi trường và dataset mà chưa tải model/chạy train:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\hy3dshape\scripts\train_windows_rtx3090_lora.ps1 `
  -SmokeTest -PreflightOnly
```

Launcher sẽ:

1. Kiểm tra Python, PyTorch CUDA, BF16, GPU và PEFT.
2. Validate bộ mini dataset.
3. Chạy trực tiếp một process/một GPU, không dùng Bash, DeepSpeed hoặc NCCL.
4. Lưu config snapshot và adapter cuối vào output.

Smoke test mặc định dùng output riêng `lora_rtx3090_windows_smoke`, nên không trộn artefact với lần train pilot.

Kết quả smoke test đã xác nhận trên máy này ngày 2026-07-16:

- Preflight đạt với PyTorch `2.5.1+cu124`, CUDA runtime `12.4`, BF16 và `torch-cluster` CUDA FPS.
- Dataset mini đạt: 8 object, 192 ảnh RGBA và 8 surface archive.
- LoRA có 5.160.960 / 3.055.919.680 tham số trainable (0,1689%).
- Hai optimizer step hoàn tất trong khoảng 6,7 giây; loss từng batch là `1,13` và `1,23`, đều hữu hạn và không OOM.
- Kiểm tra trong bộ nhớ xác nhận 200 tensor trainable đã đổi, tổng cộng 2.752.567 phần tử.
- Adapter rank 8 gồm 336 tensor FP32/5.160.960 phần tử đã được đọc lại thành công từ `hy3dshape/output_folder/dit/lora_rtx3090_windows_smoke/lora/final`.
- Toàn bộ 2.752.512 phần tử `lora_B` trong file đã lưu đều khác 0; `max_abs` khoảng `4,9974e-7`, chứng minh adapter không còn là trạng thái khởi tạo.
- Nạp lại chính adapter `final` vào model pretrained giữ chính xác 336/336 tensor và dtype FP32, xác nhận luồng resume không làm đổi trọng số.

Lỗi `AssertionError` từng xuất hiện trước batch đầu là do phần model summary của Lightning 1.9.5 bị tràn số khi đếm hơn 3 tỷ tham số trên Windows, không phải model hoặc LoRA bị hỏng. Source hiện truyền `enable_model_summary: false`; thay đổi này chỉ tắt bảng thống kê tự động và không tắt huấn luyện.

Smoke test đầu tiên trước bản sửa FP32 tuy đi hết control flow nhưng `lora_B` vẫn toàn số 0 do LR warm-up `5e-11` bị underflow trong FP16. Output đó đã bị xóa. Source hiện giữ LoRA/optimizer ở FP32 và sẽ chủ động báo lỗi nếu smoke test kết thúc mà không có giá trị trainable nào thay đổi.

Theo dõi VRAM bằng cửa sổ PowerShell khác:

```powershell
nvidia-smi -l 1
```

Chỉ chuyển sang chạy dài khi smoke test hoàn tất mà không OOM, loss hữu hạn và thư mục adapter có đủ hai file PEFT.

## 9. Chạy pilot 10.000 step

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\hy3dshape\scripts\train_windows_rtx3090_lora.ps1
```

Output mặc định:

```text
hy3dshape/output_folder/dit/lora_rtx3090_windows/
├── training_config_source.yaml
├── training_config_effective.yaml
├── log/
└── lora/
    ├── step_00000500/
    ├── step_00001000/
    └── final/
```

`training_config_source.yaml` là bản YAML đầu vào; `training_config_effective.yaml` ghi lại cấu hình thực sau khi áp dụng đường dẫn dataset và tùy chọn smoke test.

Có thể truyền đường dẫn riêng:

```powershell
.\hy3dshape\scripts\train_windows_rtx3090_lora.ps1 `
  -TrainDataset "E:\du_lieu_3d\train\preprocessed" `
  -ValDataset "E:\du_lieu_3d\val\preprocessed" `
  -OutputDir "E:\hunyuan_outputs\pilot_01"
```

Launcher validate hai đường dẫn và truyền chúng vào `train_data_list`/`val_data_list`, vì vậy không cần sửa YAML. Với bộ mini mặc định, train và validation cùng trỏ vào 8 object chỉ để smoke/overfit; dữ liệu thật nên tách UID giữa hai tập.

Launcher từ chối thư mục output không rỗng để tránh trộn adapter/log cũ. Chỉ dùng `-AllowExistingOutput` khi chủ động chấp nhận việc ghi đè/trộn artefact; phương án an toàn hơn là chọn `-OutputDir` mới.

## 10. Tiếp tục từ adapter

Để tiếp tục từ trọng số LoRA đã lưu, sửa trong YAML:

```yaml
lora_config:
  adapter_path: "E:/hunyuan_outputs/pilot_01/lora/step_00001000"
```

Cách này nạp lại trọng số adapter nhưng optimizer, scheduler và global step bắt đầu lại. Full Lightning checkpoint đang tắt để tránh file rất lớn. Nếu sau này cần resume chính xác toàn bộ trạng thái, phải bật `save_full_checkpoint` và dự trù nhiều dung lượng ổ đĩa.

Source hiện bảo toàn chính xác tensor FP32 khi nạp `adapter_path`; phép kiểm tra round-trip trên adapter smoke cuối đã khớp 336/336 tensor. Dù vậy, đây vẫn là tiếp tục từ **trọng số LoRA**, không phải resume optimizer state/global step.

## 11. Dùng adapter khi inference

```python
import sys
import torch

sys.path.insert(0, "./hy3dshape")  # Khi chạy từ thư mục gốc repository
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1",
    device="cuda",
    dtype=torch.float16,
)
pipeline.load_lora_adapter(
    r"E:\hunyuan_outputs\pilot_01\lora\final",
    merge=True,
)

meshes = pipeline(image="input.png")
meshes[0].export("output.glb")
```

## 12. Nếu smoke test vẫn OOM

Thực hiện theo thứ tự:

1. Đóng Gradio, trình duyệt, Blender và mọi chương trình đang dùng GPU.
2. Đặt `num_workers: 0` và `val_num_workers: 0` để loại trừ lỗi worker; việc này chủ yếu giảm RAM/độ phức tạp, không giảm nhiều VRAM.
3. Giữ batch size 1 và tắt validation trong smoke test.
4. Không kỳ vọng tăng `update_every` làm giảm VRAM đỉnh; nó chỉ thay đổi effective batch.
5. Nếu vẫn OOM, bước phát triển tiếp theo là cache trước ShapeVAE latents/DINO embeddings hoặc offload model đóng băng. Không giảm latent tokens tùy tiện.

## 13. Giới hạn còn lại trên Windows

- Đây chưa phải pipeline end-to-end từ mesh thô sang dữ liệu train trên Windows. `hy3dshape/tools/pipeline.sh` vẫn là Bash và dùng Blender path kiểu Linux. Bộ mini preprocessed có thể dùng ngay; dữ liệu GLB/OBJ mới cần một bước riêng để chuyển Blender/render/watertight sang PowerShell hoặc chạy thủ công.
- Native Windows có thể gặp dependency/CUDA kernel chưa được bộ source gốc kiểm thử. Nếu blocker nằm ở thư viện chỉ hỗ trợ Linux, Docker Desktop hoặc WSL2 vẫn là đường chuyển tiếp sau này.
- Phần Paint/PBR không được train trong config này.
- Hãy đọc đầy đủ `TENCENT HUNYUAN 3D 2.1 COMMUNITY LICENSE AGREEMENT` trong file `LICENSE` và license của các thành phần thứ ba. Không nên diễn giải license này đơn giản là “phi thương mại”; nó có điều kiện riêng về lãnh thổ, sử dụng, phân phối và hoạt động thương mại.
