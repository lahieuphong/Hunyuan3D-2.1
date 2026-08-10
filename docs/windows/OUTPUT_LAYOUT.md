# Bố cục dữ liệu đầu ra trên Windows

Tất cả dữ liệu do các workflow Windows của repository tạo ra được tập trung dưới một thư mục quản lý:

```text
hy3dshape/output_folder/
└── webui/
    ├── generations/       # Mỗi lượt tạo 3D là một thư mục UUID
    ├── training/          # Checkpoint và adapter LoRA; tạo khi cần
    ├── inference/         # GLB từ launcher CLI; tạo khi cần
    ├── quality_tests/     # Artefact QA; tạo khi chạy công cụ QA
    ├── projects/          # Gói công việc riêng; chỉ có khi được lưu
    ├── logs/              # stdout/stderr và log lịch sử
    ├── runtime/           # PID và trạng thái tiến trình tạm thời
    └── archive/           # Chỉ xuất hiện khi migration có dữ liệu cũ
```

Tên `webui` ở đây là **storage root được quản lý chung**, không có nghĩa mọi thư mục con đều được public. Server chỉ mount `webui/generations` tại URL `/static`; checkpoint, log, QA và project không thể tải qua route này.

WebUI chỉ chủ động tạo `generations`, `logs` và `runtime`. Các nhánh còn lại được tạo theo nhu cầu, vì vậy sau khi dọn sạch chúng sẽ không tự xuất hiện lại dưới dạng thư mục rỗng.

## Chuyển bố cục cũ

Trước tiên dừng WebUI:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File ".\hy3dshape\scripts\start_windows_multiview_webui.ps1" `
    -Stop
```

Xem trước chính xác các thao tác, không thay đổi file:

```powershell
& .\.venv-win\Scripts\python.exe `
    .\hy3dshape\scripts\migrate_output_layout.py
```

Nếu báo cáo đúng, áp dụng migration:

```powershell
& .\.venv-win\Scripts\python.exe `
    .\hy3dshape\scripts\migrate_output_layout.py `
    --apply
```

Migration chỉ di chuyển nguyên thư mục trên cùng ổ đĩa, không ghi đè, không trộn thư mục và không xóa bản trùng. Có thể chạy lại an toàn; bố cục đã cập nhật sẽ được bỏ qua.

Khi launcher khởi động, các generation UUID kiểu cũ còn nằm trực tiếp trong `webui` cũng được chuyển an toàn vào `webui/generations`. URL `?generation=<uuid>`, lịch sử và `/generation-viewer/<uuid>` không thay đổi.

## Quy tắc bảo trì

- Không ghi checkpoint, log hay dữ liệu thử nghiệm trực tiếp vào `generations`.
- Không tự xóa nội dung `archive`; hãy kiểm tra thủ công trước khi dọn.
- Không đổi tên thư mục UUID bên trong `generations`.
- Sao lưu cả `webui` nếu cần chuyển toàn bộ lịch sử và các workflow sang máy khác.
