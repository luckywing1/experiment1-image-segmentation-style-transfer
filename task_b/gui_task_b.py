"""
task_b/gui_task_b.py
任务B Tkinter 交互界面 —— 实时图像风格迁移
- 三联预览：原图 / 风格图 / 迁移结果
- 风格强度滑块（0~100）
- 内置风格图快速切换
- 摄像头实时风格迁移
- 批量处理文件夹
- 底部状态栏：推理耗时 / GPU 显存
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

# 项目根目录
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from task_b.model.adain import DEFAULT_VGG_NORMALISED_PATH, download_decoder, download_vgg_normalised
from task_b.style_transfer import StyleTransfer, DEFAULT_DECODER_PATH, DEFAULT_VGG_PATH
from task_b.gui_mask_b import MaskStyleWindow
# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
PREVIEW_W = 380
PREVIEW_H = 280
WINDOW_TITLE = "任务B：实时图像风格迁移"
BG_COLOR = "#13131f"
PANEL_COLOR = "#1e1e2e"
CARD_COLOR = "#2a2a3e"
ACCENT = "#cba6f7"
TEXT_COLOR = "#cdd6f4"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BTN_ACTIVE = "#a6adc8"

# 内置风格图目录
_STYLE_DIR = _ROOT / "assets" / "style_images"

# 内置风格名（用户将图片放入 assets/style_images/ 目录即可）
_BUILTIN_STYLE_NAMES = []


def _scan_builtin_styles():
    global _BUILTIN_STYLE_NAMES
    if _STYLE_DIR.exists():
        _BUILTIN_STYLE_NAMES = [
            f.name for f in sorted(_STYLE_DIR.iterdir())
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]


def resize_for_preview(img: Image.Image, w=PREVIEW_W, h=PREVIEW_H) -> ImageTk.PhotoImage:
    img = img.copy()
    img.thumbnail((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def placeholder_image(w=PREVIEW_W, h=PREVIEW_H, text="未加载") -> ImageTk.PhotoImage:
    arr = np.full((h, w, 3), 40, dtype=np.uint8)
    pil = Image.fromarray(arr)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    draw.text((w // 2 - 20, h // 2 - 8), text, fill=(120, 120, 120))
    return ImageTk.PhotoImage(pil)


# ─────────────────────────────────────────────────────────────
# 主应用
# ─────────────────────────────────────────────────────────────
class TaskBApp(tk.Tk):
    def __init__(self):
        super().__init__()
        _scan_builtin_styles()
        self.title(WINDOW_TITLE)
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        # 数据状态
        self._content_pil: Image.Image = None
        self._style_pil: Image.Image = None
        self._result_pil: Image.Image = None

        # 推理引擎（延迟加载）
        self._st = StyleTransfer()
        self._st_loaded = False
        self._decoder_path = DEFAULT_DECODER_PATH
        self._vgg_path = DEFAULT_VGG_NORMALISED_PATH

        # 摄像头状态
        self._cam_running = False
        self._cam_thread = None
        self._cam_cap = None

        # 批量处理状态
        self._batch_running = False

        # PhotoImage 引用防 GC
        self._photo_refs = {}

        self._build_ui()
        self._check_decoder_on_startup()

    # ─────────────────────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 标题栏 ──
        title_bar = tk.Frame(self, bg=BG_COLOR)
        title_bar.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(
            title_bar, text="实验一 · 任务B  |  实时图像风格迁移（AdaIN）",
            bg=BG_COLOR, fg=ACCENT, font=("微软雅黑", 13, "bold")
        ).pack(side="left")

        # ── 三联预览区 ──
        preview_frame = tk.Frame(self, bg=BG_COLOR)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=8)

        panel_configs = [
            ("原图（内容图）", "content"),
            ("风格参考图",     "style"),
            ("迁移结果",       "result"),
        ]
        self._preview_widgets = {}
        for col, (label_text, key) in enumerate(panel_configs):
            panel = tk.Frame(preview_frame, bg=CARD_COLOR,
                             bd=1, relief="solid")
            panel.grid(row=0, column=col, padx=5, pady=4, sticky="nsew")
            preview_frame.columnconfigure(col, weight=1)

            tk.Label(panel, text=label_text, bg=CARD_COLOR, fg=TEXT_COLOR,
                     font=("微软雅黑", 9, "bold")).pack(pady=(4, 0))

            img_lbl = tk.Label(panel, bg=CARD_COLOR,
                               image=placeholder_image(text=label_text))
            img_lbl.pack(padx=4, pady=4)
            self._preview_widgets[key] = img_lbl

        # ── 控制面板 ──
        ctrl = tk.Frame(self, bg=PANEL_COLOR, bd=1, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=4)

        # 行1：加载图像
        row1 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row1.pack(fill="x", padx=8, pady=6)

        self._make_btn(row1, "加载内容图", self._load_content).pack(
            side="left", padx=4)
        self._content_lbl = tk.Label(row1, text="未选择", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                     font=("微软雅黑", 8), width=22, anchor="w")
        self._content_lbl.pack(side="left", padx=4)

        self._make_btn(row1, "加载风格图", self._load_style).pack(
            side="left", padx=4)
        self._style_lbl = tk.Label(row1, text="未选择", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                   font=("微软雅黑", 8), width=22, anchor="w")
        self._style_lbl.pack(side="left", padx=4)

        # 行2：内置风格选择
        row2 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row2.pack(fill="x", padx=8, pady=2)

        tk.Label(row2, text="内置风格：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")

        self._builtin_var = tk.StringVar(value="（自定义）")
        options = ["（自定义）"] + _BUILTIN_STYLE_NAMES
        self._builtin_combo = ttk.Combobox(
            row2, textvariable=self._builtin_var,
            values=options, state="readonly", width=22
        )
        self._builtin_combo.pack(side="left", padx=(0, 8))
        self._builtin_combo.bind(
            "<<ComboboxSelected>>", self._on_builtin_style_select)

        tk.Label(row2, text="  decoder.pth：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")
        self._decoder_lbl = tk.Label(row2, text=self._short_path(self._decoder_path),
                                     bg=PANEL_COLOR, fg=YELLOW,
                                     font=("微软雅黑", 8), width=30, anchor="w")
        self._decoder_lbl.pack(side="left", padx=4)
        self._make_btn(row2, "选择权重文件", self._choose_decoder).pack(
            side="left", padx=4)
        self._make_btn(row2, "自动下载", self._download_decoder).pack(
            side="left", padx=4)

        # 行3：vgg_normalised.pth 状态
        row2b = tk.Frame(ctrl, bg=PANEL_COLOR)
        row2b.pack(fill="x", padx=8, pady=2)

        tk.Label(row2b, text="vgg_normalised.pth：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")
        self._vgg_lbl = tk.Label(row2b, text=self._short_path(self._vgg_path),
                                 bg=PANEL_COLOR, fg=YELLOW,
                                 font=("微软雅黑", 8), width=30, anchor="w")
        self._vgg_lbl.pack(side="left", padx=4)
        self._make_btn(row2b, "选择文件", self._choose_vgg).pack(
            side="left", padx=4)
        self._make_btn(row2b, "自动下载", self._download_vgg).pack(
            side="left", padx=4)

        # 行3：风格强度参数
        row3 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row3.pack(fill="x", padx=8, pady=4)

        tk.Label(row3, text="风格强度：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")
        self._alpha_var = tk.IntVar(value=80)
        tk.Scale(
            row3, from_=0, to=100, orient="horizontal",
            variable=self._alpha_var, bg=PANEL_COLOR, fg=TEXT_COLOR,
            highlightthickness=0, troughcolor="#444",
            length=200, font=("微软雅黑", 8)
        ).pack(side="left", padx=(0, 4))
        tk.Label(row3, textvariable=self._alpha_var, bg=PANEL_COLOR,
                 fg=TEXT_COLOR, width=3, font=("微软雅黑", 9)).pack(side="left")

        tk.Label(row3, text="  内容图最大边（px）：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left", padx=(16, 0))
        self._size_var = tk.IntVar(value=512)
        ttk.Combobox(
            row3, textvariable=self._size_var,
            values=[256, 384, 512, 640, 768], state="readonly", width=6
        ).pack(side="left", padx=4)

        # 行4：操作按钮
        row4 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row4.pack(fill="x", padx=8, pady=(4, 8))

        self._btn_run = self._make_btn(
            row4, "▶ 风格迁移", self._run_single, accent=True)
        self._btn_run.pack(side="left", padx=4)

        self._btn_cam = self._make_btn(row4, "开启摄像头", self._toggle_camera)
        self._btn_cam.pack(side="left", padx=4)

        self._btn_batch = self._make_btn(row4, "批量处理文件夹", self._run_batch)
        self._btn_batch.pack(side="left", padx=4)

        self._btn_save = self._make_btn(row4, "保存结果", self._save_result)
        self._btn_save.pack(side="left", padx=4)

        self._make_btn(row4, "🎨 掩码分区风格", self._open_mask_window).pack(
            side="left", padx=4)

        # ── 状态栏 ──
        self._status_var = tk.StringVar(
            value="就绪  |  请加载内容图和风格图（权重文件首次运行时自动下载）")
        status_bar = tk.Frame(self, bg="#0d0d1a")
        status_bar.pack(fill="x", side="bottom")

        tk.Label(
            status_bar, textvariable=self._status_var,
            bg="#0d0d1a", fg="#89b4fa",
            font=("Consolas", 9), anchor="w", padx=10
        ).pack(side="left", fill="x", expand=True)

        self._vram_var = tk.StringVar(value="VRAM: --")
        tk.Label(
            status_bar, textvariable=self._vram_var,
            bg="#0d0d1a", fg=GREEN,
            font=("Consolas", 9), padx=10
        ).pack(side="right")

    def _make_btn(self, parent, text, cmd, accent=False):
        color = ACCENT if accent else "#3d3d5c"
        fg = "#1e1e2e" if accent else "white"
        return tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=fg, relief="flat",
            activebackground=BTN_ACTIVE, activeforeground="black",
            font=("微软雅黑", 9), padx=10, pady=4, cursor="hand2"
        )

    # ─────────────────────────────────────────────────────────
    # 事件处理
    # ─────────────────────────────────────────────────────────
    def _check_decoder_on_startup(self):
        if not os.path.exists(self._decoder_path):
            self._decoder_lbl.config(fg=RED)
            self._status("未找到 decoder.pth，点击[自动下载]或首次运行时自动下载")
        else:
            self._decoder_lbl.config(fg=GREEN)
        if os.path.exists(self._vgg_path):
            self._vgg_lbl.config(fg=GREEN)
        else:
            self._vgg_lbl.config(fg=YELLOW)
            self._status("未找到 vgg_normalised.pth，点击[自动下载]或首次运行时自动下载")

    def _load_content(self):
        path = filedialog.askopenfilename(
            title="选择内容图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._content_pil = Image.open(path).convert("RGB")
            self._content_lbl.config(text=os.path.basename(path))
            self._update_preview("content", self._content_pil.copy())
            self._result_pil = None
            self._set_placeholder("result")
            self._status("已加载内容图：" + os.path.basename(path))
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _load_style(self):
        path = filedialog.askopenfilename(
            title="选择风格图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._style_pil = Image.open(path).convert("RGB")
            self._style_lbl.config(text=os.path.basename(path))
            self._builtin_var.set("（自定义）")
            self._update_preview("style", self._style_pil.copy())
            self._st.reset_smoother()
            self._status("已加载风格图：" + os.path.basename(path))
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _on_builtin_style_select(self, event=None):
        name = self._builtin_var.get()
        if name == "（自定义）":
            return
        path = _STYLE_DIR / name
        try:
            self._style_pil = Image.open(str(path)).convert("RGB")
            self._style_lbl.config(text=name)
            self._update_preview("style", self._style_pil.copy())
            self._st.reset_smoother()
            self._status("已切换内置风格：" + name)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _choose_decoder(self):
        path = filedialog.askopenfilename(
            title="选择 decoder.pth",
            filetypes=[("PyTorch 权重", "*.pth *.pt"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self._decoder_path = path
        self._decoder_lbl.config(text=self._short_path(path), fg=GREEN)
        self._st_loaded = False  # 需要重新加载
        self._status("已选择权重：" + os.path.basename(path) + "，下次运行时自动加载")

    def _download_decoder(self):
        """手动触发下载 decoder.pth。"""
        if os.path.exists(self._decoder_path):
            self._status("decoder.pth 已存在，无需下载")
            self._decoder_lbl.config(fg=GREEN)
            return
        self._status("正在下载 decoder.pth（~13 MB），请稍候...")
        threading.Thread(target=self._download_decoder_worker,
                         daemon=True).start()

    def _download_decoder_worker(self):
        try:
            download_decoder(self._decoder_path)
            self.after(0, lambda: self._decoder_lbl.config(fg=GREEN))
            self._status("decoder.pth 下载完成")
        except Exception as e:
            self.after(0, lambda: self._decoder_lbl.config(fg=RED))
            self._status("下载失败：" + str(e))

    def _choose_vgg(self):
        path = filedialog.askopenfilename(
            title="选择 vgg_normalised.pth",
            filetypes=[("PyTorch 权重", "*.pth *.pt"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self._vgg_path = path
        self._vgg_lbl.config(text=self._short_path(path), fg=GREEN)
        self._st_loaded = False
        self._status("已选择 VGG 权重：" + os.path.basename(path))

    def _download_vgg(self):
        """手动触发下载 vgg_normalised.pth。"""
        if os.path.exists(self._vgg_path):
            self._status("vgg_normalised.pth 已存在，无需下载")
            self._vgg_lbl.config(fg=GREEN)
            return
        self._status("正在下载 vgg_normalised.pth（~80 MB），请稍候...")
        threading.Thread(target=self._download_vgg_worker, daemon=True).start()

    def _download_vgg_worker(self):
        try:
            download_vgg_normalised(self._vgg_path)
            self.after(0, lambda: self._vgg_lbl.config(fg=GREEN))
            self._status("vgg_normalised.pth 下载完成")
        except Exception as e:
            self.after(0, lambda: self._vgg_lbl.config(fg=RED))
            self._status("下载失败：" + str(e))

    def _ensure_model_loaded(self) -> bool:
        if self._st_loaded:
            return True
        try:
            self._st.max_content_size = self._size_var.get()
            self._st.load(self._decoder_path,
                          vgg_normalised_path=self._vgg_path)
            self._st_loaded = True
            return True
        except FileNotFoundError as e:
            messagebox.showerror("权重文件缺失", str(e))
            return False
        except Exception as e:
            messagebox.showerror("模型加载失败", str(e))
            return False

    def _run_single(self):
        if self._content_pil is None:
            messagebox.showwarning("提示", "请先加载内容图！")
            return
        if self._style_pil is None:
            messagebox.showwarning("提示", "请先加载风格图！")
            return
        if self._cam_running:
            messagebox.showwarning("提示", "请先关闭摄像头模式！")
            return
        self._btn_run.config(state="disabled", text="处理中...")
        self._status("正在推理，请稍候...")
        threading.Thread(target=self._run_single_worker, daemon=True).start()

    def _run_single_worker(self):
        try:
            if not self._ensure_model_loaded():
                return
            alpha = self._alpha_var.get() / 100.0
            result, ms = self._st.transfer(
                self._content_pil, self._style_pil, alpha=alpha
            )
            self._result_pil = result
            self._update_preview("result", result.copy())
            vram = self._st.get_vram_mb()
            self._status(
                "完成  |  推理耗时 %.1f ms  |  总耗时 %.1f ms" % (
                    ms, self._st.last_total_ms)
            )
            self._vram_var.set("VRAM: %.0f MB" %
                               vram if vram > 0 else "VRAM: CPU 模式")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("推理错误", str(e)))
            self._status("错误：" + str(e))
        finally:
            self.after(0, lambda: self._btn_run.config(
                state="normal", text="▶ 风格迁移"
            ))

    # ── 摄像头 ──
    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self._style_pil is None:
            messagebox.showwarning("提示", "请先加载风格图！")
            return
        if not self._ensure_model_loaded():
            return

        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头！请检查摄像头是否可用。")
            return

        self._cam_cap = cap
        self._cam_running = True
        self._btn_cam.config(text="关闭摄像头", bg=RED, fg="white")
        self._st.reset_smoother()
        self._cam_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self._status("摄像头模式已开启  |  点击[关闭摄像头]退出")

    def _stop_camera(self):
        self._cam_running = False
        if self._cam_cap:
            self._cam_cap.release()
            self._cam_cap = None
        self._btn_cam.config(text="开启摄像头", bg="#3d3d5c", fg="white")
        self._status("摄像头模式已关闭")

    def _camera_loop(self):
        import cv2
        while self._cam_running:
            ret, frame = self._cam_cap.read()
            if not ret:
                break

            # 实时读取 alpha（支持滑块调节）
            alpha = self._alpha_var.get() / 100.0

            result_bgr, ms = self._st.process_camera_frame(
                frame, self._style_pil, alpha=alpha
            )

            # 内容图预览（摄像头当前帧）
            content_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content_pil = Image.fromarray(content_rgb)
            self._update_preview("content", content_pil)

            # 结果预览
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            self._update_preview("result", result_pil)
            self._result_pil = result_pil

            vram = self._st.get_vram_mb()
            self._status("摄像头模式  |  推理 %.1f ms/帧  |  alpha=%.2f" % (ms, alpha))
            self._vram_var.set("VRAM: %.0f MB" %
                               vram if vram > 0 else "VRAM: CPU 模式")

        self.after(0, self._stop_camera)

    # ── 批量处理 ──
    def _run_batch(self):
        if self._style_pil is None:
            messagebox.showwarning("提示", "请先加载风格图！")
            return
        if self._cam_running:
            messagebox.showwarning("提示", "请先关闭摄像头模式！")
            return

        input_dir = filedialog.askdirectory(title="选择内容图像文件夹")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="选择输出文件夹")
        if not output_dir:
            return

        # 临时保存风格图
        import tempfile
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._style_pil.save(tf.name)
        style_tmp = tf.name
        tf.close()

        alpha = self._alpha_var.get() / 100.0
        self._btn_batch.config(state="disabled", text="批量处理中...")
        threading.Thread(
            target=self._batch_worker,
            args=(input_dir, style_tmp, output_dir, alpha),
            daemon=True
        ).start()

    def _batch_worker(self, input_dir, style_tmp, output_dir, alpha):
        try:
            if not self._ensure_model_loaded():
                return

            def progress(cur, total, name):
                self._status("批量处理 %d/%d：%s" % (cur, total, name))

            paths = self._st.batch_transfer(
                input_dir, style_tmp, output_dir, alpha=alpha,
                make_grid=True, progress_callback=progress
            )
            self._status("批量处理完成，共 %d 个文件  ->  输出目录：%s" %
                         (len(paths), output_dir))
            self.after(0, lambda: messagebox.showinfo(
                "完成",
                "批量处理完成！\n共处理 %d 张图像\n输出目录：%s" % (len(paths), output_dir)
            ))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("批量处理错误", str(e)))
            self._status("错误：" + str(e))
        finally:
            import os
            try:
                os.unlink(style_tmp)
            except Exception:
                pass
            self.after(0, lambda: self._btn_batch.config(
                state="normal", text="批量处理文件夹"
            ))

    # ── 保存 ──
    def _save_result(self):
        if self._result_pil is None:
            messagebox.showwarning("提示", "请先运行风格迁移！")
            return
        path = filedialog.asksaveasfilename(
            title="保存迁移结果",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("所有文件", "*.*")]
        )
        if path:
            self._result_pil.save(path)
            self._status("结果已保存：" + os.path.basename(path))

    # ── 掩码分区风格迁移 ──
    def _open_mask_window(self):
        """打开掩码分区风格迁移窗口，共享已加载的模型。"""
        # 确保模型已加载，否则先触发加载（与单张风格迁移共用实例）
        if not self._ensure_model_loaded():
            return
        win = MaskStyleWindow(self, self._st)
        win.grab_set()  # 模态对话框

    # ─────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────
    def _update_preview(self, key: str, pil_img: Image.Image):
        photo = resize_for_preview(pil_img)
        self._photo_refs[key] = photo
        self.after(0, lambda: self._preview_widgets[key].config(image=photo))

    def _set_placeholder(self, key: str):
        photo = placeholder_image(text=key)
        self._photo_refs["ph_" + key] = photo
        self.after(0, lambda: self._preview_widgets[key].config(image=photo))

    def _status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    @staticmethod
    def _short_path(p: str, max_len: int = 45) -> str:
        if len(p) <= max_len:
            return p
        return "..." + p[-(max_len - 3):]

    def destroy(self):
        self._cam_running = False
        if self._cam_cap:
            self._cam_cap.release()
        super().destroy()


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TaskBApp()
    app.mainloop()
