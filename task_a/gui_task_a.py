"""
task_a/gui_task_a.py
任务A Tkinter 交互界面
- 四联预览：原图 / 分割掩码 / 提取前景 / 合成结果
- 支持：类别选择、羽化强度、融合方式（Alpha/泊松）、背景图加载、结果保存
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# 将项目根目录加入 sys.path，确保跨目录导入正常
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from task_a.segmentation import Segmentor, CLASS_NAMES_ZH
from task_a.compositing import (
    extract_foreground,
    alpha_blend,
    poisson_blend,
    overlay_rgba_on_checker,
    mask_to_display,
)

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
PREVIEW_W = 320
PREVIEW_H = 240
WINDOW_TITLE = "任务A：图像分割与前景合成"
BG_COLOR = "#1e1e2e"
PANEL_COLOR = "#2a2a3e"
ACCENT = "#7c6af7"
TEXT_COLOR = "#cdd6f4"
BTN_ACTIVE = "#a6adc8"


def resize_for_preview(img: Image.Image, w=PREVIEW_W, h=PREVIEW_H) -> ImageTk.PhotoImage:
    """等比缩放图像并转换为 Tkinter 可用的 PhotoImage。"""
    img.thumbnail((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def placeholder_image(w=PREVIEW_W, h=PREVIEW_H, text="未加载") -> ImageTk.PhotoImage:
    """生成灰色占位图。"""
    arr = np.full((h, w, 3), 50, dtype=np.uint8)
    pil = Image.fromarray(arr)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    draw.text((w // 2 - 20, h // 2 - 8), text, fill=(150, 150, 150))
    return ImageTk.PhotoImage(pil)


# ─────────────────────────────────────────────────────────────
# 主应用
# ─────────────────────────────────────────────────────────────
class TaskAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        # 数据状态
        self._orig_pil: Image.Image = None
        self._bg_pil: Image.Image = None
        self._mask_np: np.ndarray = None
        self._fg_rgba: Image.Image = None
        self._result_pil: Image.Image = None

        # 分割器（延迟加载）
        self._segmentor = Segmentor()

        # 保存 PhotoImage 引用，防止被 GC
        self._photo_refs = {}

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 顶部标题 ──
        title_bar = tk.Frame(self, bg=BG_COLOR)
        title_bar.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(
            title_bar, text="实验一 · 任务A  |  图像分割与前景合成",
            bg=BG_COLOR, fg=ACCENT, font=("微软雅黑", 13, "bold")
        ).pack(side="left")

        # ── 四联预览区 ──
        preview_frame = tk.Frame(self, bg=BG_COLOR)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=8)

        labels_texts = ["原图", "分割掩码", "提取前景（含透明）", "合成结果"]
        self._preview_labels = []
        self._preview_widgets = []

        for col, text in enumerate(labels_texts):
            panel = tk.Frame(preview_frame, bg=PANEL_COLOR, bd=1, relief="solid")
            panel.grid(row=0, column=col, padx=4, pady=4, sticky="nsew")
            preview_frame.columnconfigure(col, weight=1)

            tk.Label(panel, text=text, bg=PANEL_COLOR, fg=TEXT_COLOR,
                     font=("微软雅黑", 9, "bold")).pack(pady=(4, 0))

            img_lbl = tk.Label(panel, bg=PANEL_COLOR,
                               image=placeholder_image(text=text))
            img_lbl.pack(padx=4, pady=4)
            self._preview_widgets.append(img_lbl)

        # ── 控制区 ──
        ctrl = tk.Frame(self, bg=PANEL_COLOR, bd=1, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=4)

        # 行1：加载图像
        row1 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row1.pack(fill="x", padx=8, pady=6)

        self._btn_load_orig = self._make_btn(row1, "📂 加载原图", self._load_orig)
        self._btn_load_orig.pack(side="left", padx=4)

        self._orig_label = tk.Label(row1, text="未选择", bg=PANEL_COLOR,
                                    fg=TEXT_COLOR, font=("微软雅黑", 8), width=30,
                                    anchor="w")
        self._orig_label.pack(side="left", padx=4)

        self._btn_load_bg = self._make_btn(row1, "🖼 加载背景图", self._load_bg)
        self._btn_load_bg.pack(side="left", padx=4)

        self._bg_label = tk.Label(row1, text="未选择（将使用纯色背景）",
                                  bg=PANEL_COLOR, fg=TEXT_COLOR,
                                  font=("微软雅黑", 8), width=30, anchor="w")
        self._bg_label.pack(side="left", padx=4)

        # 行2：参数设置
        row2 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row2.pack(fill="x", padx=8, pady=4)

        tk.Label(row2, text="目标类别：", bg=PANEL_COLOR,
                 fg=TEXT_COLOR, font=("微软雅黑", 9)).pack(side="left")
        self._class_var = tk.StringVar(value="人")
        cls_combo = ttk.Combobox(
            row2, textvariable=self._class_var,
            values=CLASS_NAMES_ZH[1:],  # 排除"背景"
            state="readonly", width=10
        )
        cls_combo.pack(side="left", padx=(0, 16))

        tk.Label(row2, text="羽化强度：", bg=PANEL_COLOR,
                 fg=TEXT_COLOR, font=("微软雅黑", 9)).pack(side="left")
        self._feather_var = tk.IntVar(value=12)
        feather_scale = tk.Scale(
            row2, from_=0, to=40, orient="horizontal",
            variable=self._feather_var, bg=PANEL_COLOR, fg=TEXT_COLOR,
            highlightthickness=0, troughcolor="#444", length=140,
            font=("微软雅黑", 8)
        )
        feather_scale.pack(side="left", padx=(0, 16))

        tk.Label(row2, text="融合方式：", bg=PANEL_COLOR,
                 fg=TEXT_COLOR, font=("微软雅黑", 9)).pack(side="left")
        self._blend_var = tk.StringVar(value="Alpha 融合")
        for mode in ("Alpha 融合", "泊松融合"):
            tk.Radiobutton(
                row2, text=mode, variable=self._blend_var, value=mode,
                bg=PANEL_COLOR, fg=TEXT_COLOR, selectcolor=PANEL_COLOR,
                activebackground=PANEL_COLOR, font=("微软雅黑", 9)
            ).pack(side="left", padx=4)

        # 行3：操作按钮
        row3 = tk.Frame(ctrl, bg=PANEL_COLOR)
        row3.pack(fill="x", padx=8, pady=(4, 8))

        self._btn_run = self._make_btn(row3, "▶ 运行分割与合成", self._run, accent=True)
        self._btn_run.pack(side="left", padx=4)

        self._btn_save = self._make_btn(row3, "💾 保存合成结果", self._save_result)
        self._btn_save.pack(side="left", padx=4)

        self._btn_save_fg = self._make_btn(row3, "💾 保存前景(PNG)", self._save_fg)
        self._btn_save_fg.pack(side="left", padx=4)

        # 状态栏
        self._status_var = tk.StringVar(value="就绪  |  请先加载原图")
        status_bar = tk.Label(
            self, textvariable=self._status_var,
            bg="#13131f", fg="#89b4fa",
            font=("Consolas", 9), anchor="w", padx=10
        )
        status_bar.pack(fill="x", side="bottom")

    def _make_btn(self, parent, text, cmd, accent=False):
        color = ACCENT if accent else "#3d3d5c"
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white", relief="flat",
            activebackground=BTN_ACTIVE, activeforeground="black",
            font=("微软雅黑", 9), padx=10, pady=4, cursor="hand2"
        )
        return btn

    # ─────────────────────────────────────────────────────────
    # 事件处理
    # ─────────────────────────────────────────────────────────
    def _load_orig(self):
        path = filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._orig_pil = Image.open(path).convert("RGB")
            self._orig_label.config(text=os.path.basename(path))
            self._update_preview(0, self._orig_pil.copy())
            # 重置中间结果
            self._mask_np = None
            self._fg_rgba = None
            self._result_pil = None
            for i in (1, 2, 3):
                self._set_placeholder(i)
            self._status("已加载原图：" + os.path.basename(path))
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _load_bg(self):
        path = filedialog.askopenfilename(
            title="选择背景图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._bg_pil = Image.open(path).convert("RGB")
            self._bg_label.config(text=os.path.basename(path))
            self._status("已加载背景图：" + os.path.basename(path))
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _run(self):
        if self._orig_pil is None:
            messagebox.showwarning("提示", "请先加载原图！")
            return
        self._btn_run.config(state="disabled", text="运行中...")
        self._status("正在分割，请稍候…（首次运行需下载模型约 300 MB）")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            import time
            cls_name = self._class_var.get()
            feather_r = self._feather_var.get()
            blend_mode = self._blend_var.get()

            # ── 分割 ──
            t0 = time.time()
            self._mask_np = self._segmentor.segment(self._orig_pil, cls_name)
            seg_ms = (time.time() - t0) * 1000

            mask_display = mask_to_display(self._mask_np)
            self._update_preview(1, mask_display)

            # ── 提取前景 ──
            self._fg_rgba = extract_foreground(
                self._orig_pil, self._mask_np, feather_radius=feather_r
            )
            fg_display = overlay_rgba_on_checker(self._fg_rgba)
            self._update_preview(2, fg_display)

            # ── 合成 ──
            if self._bg_pil is not None:
                bg = self._bg_pil.copy()
                # 等比缩放背景到原图大小
                bg = bg.resize(self._orig_pil.size, Image.LANCZOS)
            else:
                # 纯白背景
                bg = Image.new("RGB", self._orig_pil.size, (240, 240, 240))

            t1 = time.time()
            if blend_mode == "泊松融合":
                self._result_pil = poisson_blend(
                    self._orig_pil, bg, self._mask_np
                )
            else:
                self._result_pil = alpha_blend(
                    self._orig_pil, bg, self._mask_np, feather_radius=feather_r
                )
            blend_ms = (time.time() - t1) * 1000

            self._update_preview(3, self._result_pil.copy())
            self._status(
                f"完成  |  分割耗时 {seg_ms:.0f} ms  |  合成耗时 {blend_ms:.0f} ms"
                f"  |  类别={cls_name}  融合={blend_mode}"
            )

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("运行错误", str(e)))
            self._status(f"错误：{e}")
        finally:
            self.after(0, lambda: self._btn_run.config(
                state="normal", text="▶ 运行分割与合成"
            ))

    def _save_result(self):
        if self._result_pil is None:
            messagebox.showwarning("提示", "请先运行分割与合成！")
            return
        path = filedialog.asksaveasfilename(
            title="保存合成结果",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("所有文件", "*.*")]
        )
        if path:
            self._result_pil.save(path)
            self._status(f"合成结果已保存：{os.path.basename(path)}")

    def _save_fg(self):
        if self._fg_rgba is None:
            messagebox.showwarning("提示", "请先运行分割！")
            return
        path = filedialog.asksaveasfilename(
            title="保存前景（RGBA PNG）",
            defaultextension=".png",
            filetypes=[("PNG（含透明通道）", "*.png")]
        )
        if path:
            self._fg_rgba.save(path)
            self._status(f"前景已保存：{os.path.basename(path)}")

    # ─────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────
    def _update_preview(self, idx: int, pil_img: Image.Image):
        photo = resize_for_preview(pil_img)
        self._photo_refs[idx] = photo
        self.after(0, lambda: self._preview_widgets[idx].config(image=photo))

    def _set_placeholder(self, idx: int):
        photo = placeholder_image()
        self._photo_refs[f"ph_{idx}"] = photo
        self.after(0, lambda: self._preview_widgets[idx].config(image=photo))

    def _status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TaskAApp()
    app.mainloop()
