"""
task_b/gui_mask_b.py
掩码分区风格迁移窗口
- 在内容图上用画笔绘制掩码区域（每个区域可指定不同风格图）
- 运行后各区域分别风格迁移，按掩码合成最终结果
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# 项目根目录
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_b.style_transfer import StyleTransfer, DEFAULT_DECODER_PATH, DEFAULT_VGG_PATH

# ─────────────────────────────────────────────────────────────
# 颜色方案（与 gui_task_b 保持一致）
# ─────────────────────────────────────────────────────────────
BG_COLOR = "#13131f"
PANEL_COLOR = "#1e1e2e"
CARD_COLOR = "#2a2a3e"
ACCENT = "#cba6f7"
TEXT_COLOR = "#cdd6f4"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BTN_ACTIVE = "#a6adc8"

# 每个区域使用的高对比度颜色（用于画笔叠加显示）
_REGION_COLORS = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff",
    "#ff9cee", "#00c9a7", "#f5a623", "#c471ed",
]

CANVAS_W = 512
CANVAS_H = 400


# ─────────────────────────────────────────────────────────────
# 区域数据结构
# ─────────────────────────────────────────────────────────────
class Region:
    """一个掩码区域：绑定风格图 + 画笔轨迹 + 掩码位图。"""
    _id_counter = 0

    def __init__(self, color: str, name: str = None):
        Region._id_counter += 1
        self.id = Region._id_counter
        self.name = name or f"区域 {self.id}"
        self.color = color          # 画笔显示颜色
        self.style_pil: Image.Image = None
        self.style_name: str = "未选择"
        self.alpha: float = 0.8
        # 掩码：与内容图同尺寸的 PIL 'L' 图像（白=应用风格）
        self._mask: Image.Image = None
        self._mask_draw: ImageDraw.ImageDraw = None

    def init_mask(self, w: int, h: int):
        self._mask = Image.new("L", (w, h), 0)
        self._mask_draw = ImageDraw.Draw(self._mask)

    def draw_circle(self, x: int, y: int, r: int):
        """在掩码上绘制圆形笔触。"""
        if self._mask_draw:
            self._mask_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def erase_circle(self, x: int, y: int, r: int):
        """擦除掩码上的圆形区域。"""
        if self._mask_draw:
            self._mask_draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def get_mask(self) -> Image.Image:
        return self._mask

    def has_mask(self) -> bool:
        if self._mask is None:
            return False
        return np.any(np.array(self._mask) > 0)


# ─────────────────────────────────────────────────────────────
# 掩码风格迁移窗口
# ─────────────────────────────────────────────────────────────
class MaskStyleWindow(tk.Toplevel):
    def __init__(self, parent, st: StyleTransfer):
        super().__init__(parent)
        self.title("掩码分区风格迁移")
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        self._st = st
        self._content_pil: Image.Image = None   # 原始内容图
        self._result_pil: Image.Image = None

        # 画布显示比例（内容图→画布坐标变换）
        self._scale_x = 1.0
        self._scale_y = 1.0

        # 区域列表
        self._regions: list[Region] = []
        self._active_region: Region = None  # 当前正在绘制的区域

        # 画笔设置
        self._brush_size = tk.IntVar(value=20)
        self._erase_mode = tk.BooleanVar(value=False)

        # 叠加层：用于显示掩码画笔效果（RGBA）
        self._overlay: Image.Image = None
        self._photo_content = None
        self._photo_result = None

        self._drawing = False
        self._last_xy = None

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 顶部标题 ──
        tk.Label(self, text="掩码分区风格迁移",
                 bg=BG_COLOR, fg=ACCENT,
                 font=("微软雅黑", 13, "bold")).pack(pady=(10, 4))

        main = tk.Frame(self, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=4)

        # ── 左侧：画布区 ──
        left = tk.Frame(main, bg=BG_COLOR)
        left.pack(side="left", fill="both", expand=True)

        self._build_canvas(left)

        # ── 右侧：控制面板 ──
        right = tk.Frame(main, bg=PANEL_COLOR, width=280)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self._build_right_panel(right)

        # ── 底部状态栏 ──
        self._status_var = tk.StringVar(value="请先加载内容图，然后添加区域并绘制掩码")
        tk.Label(self, textvariable=self._status_var,
                 bg="#0d0d1a", fg="#89b4fa",
                 font=("Consolas", 9), anchor="w", padx=10
                 ).pack(fill="x", side="bottom", ipady=4)

    def _build_canvas(self, parent):
        tk.Label(parent, text="在内容图上绘制掩码区域",
                 bg=BG_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(anchor="w")

        canvas_frame = tk.Frame(parent, bg=CARD_COLOR,
                                bd=1, relief="solid")
        canvas_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_W, height=CANVAS_H,
            bg="#1a1a2e", cursor="crosshair",
            highlightthickness=0
        )
        self._canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # 画笔事件
        self._canvas.bind("<ButtonPress-1>",   self._on_paint_start)
        self._canvas.bind("<B1-Motion>",       self._on_paint_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_paint_end)

        # 结果预览（右下角）
        result_frame = tk.Frame(parent, bg=BG_COLOR)
        result_frame.pack(fill="x", pady=(6, 0))

        tk.Label(result_frame, text="迁移结果预览：",
                 bg=BG_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(anchor="w")

        self._result_canvas = tk.Canvas(
            result_frame,
            width=CANVAS_W, height=200,
            bg="#1a1a2e", highlightthickness=0
        )
        self._result_canvas.pack()

    def _build_right_panel(self, parent):
        pad = {"padx": 8, "pady": 4}

        # ── 加载内容图 ──
        tk.Label(parent, text="内容图",
                 bg=PANEL_COLOR, fg=ACCENT,
                 font=("微软雅黑", 10, "bold")).pack(anchor="w", **pad)

        content_row = tk.Frame(parent, bg=PANEL_COLOR)
        content_row.pack(fill="x", padx=8)
        self._make_btn(content_row, "加载内容图", self._load_content).pack(
            side="left")
        self._content_lbl = tk.Label(
            content_row, text="未加载", bg=PANEL_COLOR, fg=TEXT_COLOR,
            font=("微软雅黑", 8), anchor="w", width=18)
        self._content_lbl.pack(side="left", padx=4)

        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=6)

        # ── 区域列表 ──
        tk.Label(parent, text="掩码区域",
                 bg=PANEL_COLOR, fg=ACCENT,
                 font=("微软雅黑", 10, "bold")).pack(anchor="w", **pad)

        # 区域列表框
        list_frame = tk.Frame(parent, bg=CARD_COLOR, bd=1, relief="solid")
        list_frame.pack(fill="x", padx=8, pady=2)

        self._region_listbox = tk.Listbox(
            list_frame,
            bg=CARD_COLOR, fg=TEXT_COLOR,
            selectbackground="#3d3d5c",
            font=("微软雅黑", 9),
            height=5, activestyle="none",
            highlightthickness=0, bd=0
        )
        self._region_listbox.pack(fill="x", padx=2, pady=2)
        self._region_listbox.bind("<<ListboxSelect>>", self._on_region_select)

        # 区域操作按钮
        btn_row1 = tk.Frame(parent, bg=PANEL_COLOR)
        btn_row1.pack(fill="x", padx=8, pady=2)
        self._make_btn(btn_row1, "+ 添加区域", self._add_region, accent=True).pack(
            side="left", padx=(0, 4))
        self._make_btn(btn_row1, "删除选中", self._delete_region).pack(side="left")

        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=6)

        # ── 当前区域设置 ──
        tk.Label(parent, text="当前区域设置",
                 bg=PANEL_COLOR, fg=ACCENT,
                 font=("微软雅黑", 10, "bold")).pack(anchor="w", **pad)

        # 风格图
        style_row = tk.Frame(parent, bg=PANEL_COLOR)
        style_row.pack(fill="x", padx=8, pady=2)
        self._make_btn(style_row, "选择风格图",
                       self._choose_style).pack(side="left")
        self._style_lbl = tk.Label(
            style_row, text="未选择", bg=PANEL_COLOR, fg=YELLOW,
            font=("微软雅黑", 8), anchor="w", width=16)
        self._style_lbl.pack(side="left", padx=4)

        # 风格缩略图
        self._style_thumb_lbl = tk.Label(parent, bg=PANEL_COLOR)
        self._style_thumb_lbl.pack(pady=2)

        # Alpha
        alpha_row = tk.Frame(parent, bg=PANEL_COLOR)
        alpha_row.pack(fill="x", padx=8, pady=2)
        tk.Label(alpha_row, text="强度：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")
        self._alpha_var = tk.DoubleVar(value=0.8)
        tk.Scale(
            alpha_row, from_=0.0, to=1.0, resolution=0.05,
            orient="horizontal", variable=self._alpha_var,
            bg=PANEL_COLOR, fg=TEXT_COLOR, highlightthickness=0,
            troughcolor="#444", length=120, font=("微软雅黑", 8)
        ).pack(side="left")
        tk.Label(alpha_row, textvariable=self._alpha_var,
                 bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 8), width=4).pack(side="left")

        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=6)

        # ── 画笔设置 ──
        tk.Label(parent, text="画笔设置",
                 bg=PANEL_COLOR, fg=ACCENT,
                 font=("微软雅黑", 10, "bold")).pack(anchor="w", **pad)

        brush_row = tk.Frame(parent, bg=PANEL_COLOR)
        brush_row.pack(fill="x", padx=8, pady=2)
        tk.Label(brush_row, text="笔刷大小：", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("微软雅黑", 9)).pack(side="left")
        tk.Scale(
            brush_row, from_=5, to=80, orient="horizontal",
            variable=self._brush_size,
            bg=PANEL_COLOR, fg=TEXT_COLOR, highlightthickness=0,
            troughcolor="#444", length=110, font=("微软雅黑", 8)
        ).pack(side="left")

        erase_row = tk.Frame(parent, bg=PANEL_COLOR)
        erase_row.pack(fill="x", padx=8, pady=2)
        tk.Checkbutton(
            erase_row, text="橡皮擦模式", variable=self._erase_mode,
            bg=PANEL_COLOR, fg=TEXT_COLOR, selectcolor=CARD_COLOR,
            activebackground=PANEL_COLOR, font=("微软雅黑", 9)
        ).pack(side="left")

        self._make_btn(erase_row, "清除当前区域掩码",
                       self._clear_current_mask).pack(side="left", padx=4)

        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=8)

        # ── 执行按钮 ──
        self._btn_run = self._make_btn(
            parent, "▶ 执行掩码风格迁移", self._run, accent=True)
        self._btn_run.pack(fill="x", padx=8, pady=4)

        self._make_btn(parent, "保存结果", self._save_result).pack(
            fill="x", padx=8, pady=2)

    # ─────────────────────────────────────────────────────────
    # 辅助：按钮工厂
    # ─────────────────────────────────────────────────────────
    def _make_btn(self, parent, text, cmd, accent=False):
        color = ACCENT if accent else "#3d3d5c"
        fg = "#1e1e2e" if accent else "white"
        return tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=fg, relief="flat",
            activebackground=BTN_ACTIVE, activeforeground="black",
            font=("微软雅黑", 9), padx=8, pady=3, cursor="hand2"
        )

    # ─────────────────────────────────────────────────────────
    # 内容图加载
    # ─────────────────────────────────────────────────────────
    def _load_content(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="选择内容图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._content_pil = Image.open(path).convert("RGB")
            self._content_lbl.config(text=os.path.basename(path))
            # 重置所有区域的掩码
            W, H = self._content_pil.size
            for r in self._regions:
                r.init_mask(W, H)
            self._overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            self._refresh_canvas()
            self._status(f"已加载内容图：{os.path.basename(path)}  ({W}×{H})")
        except Exception as e:
            messagebox.showerror("加载失败", str(e), parent=self)

    # ─────────────────────────────────────────────────────────
    # 区域管理
    # ─────────────────────────────────────────────────────────
    def _add_region(self):
        if self._content_pil is None:
            messagebox.showwarning("提示", "请先加载内容图！", parent=self)
            return
        color = _REGION_COLORS[len(self._regions) % len(_REGION_COLORS)]
        region = Region(color)
        W, H = self._content_pil.size
        region.init_mask(W, H)
        self._regions.append(region)
        self._region_listbox.insert(tk.END, f"  ● {region.name}  [无风格图]")
        self._region_listbox.itemconfig(
            len(self._regions) - 1, fg=color)
        # 自动选中新区域
        self._region_listbox.selection_clear(0, tk.END)
        self._region_listbox.selection_set(len(self._regions) - 1)
        self._on_region_select()
        self._status(f"已添加 {region.name}，请在画布上绘制掩码区域")

    def _delete_region(self):
        sel = self._region_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        region = self._regions[idx]
        if messagebox.askyesno("确认", f"删除 {region.name}？", parent=self):
            self._regions.pop(idx)
            self._region_listbox.delete(idx)
            self._active_region = None
            self._style_lbl.config(text="未选择")
            self._style_thumb_lbl.config(image="")
            self._refresh_canvas()

    def _on_region_select(self, event=None):
        sel = self._region_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._active_region = self._regions[idx]
        # 同步 alpha 显示
        self._alpha_var.set(self._active_region.alpha)
        # 同步风格图显示
        if self._active_region.style_pil:
            self._style_lbl.config(
                text=self._active_region.style_name, fg=GREEN)
            self._update_style_thumb(self._active_region.style_pil)
        else:
            self._style_lbl.config(text="未选择", fg=YELLOW)
            self._style_thumb_lbl.config(image="")
        self._status(f"当前区域：{self._active_region.name}  — 在画布上绘制掩码")

    def _refresh_listbox(self):
        """刷新列表框文字。"""
        for i, r in enumerate(self._regions):
            style_info = r.style_name if r.style_pil else "无风格图"
            self._region_listbox.delete(i)
            self._region_listbox.insert(i, f"  ● {r.name}  [{style_info}]")
            self._region_listbox.itemconfig(i, fg=r.color)

    # ─────────────────────────────────────────────────────────
    # 风格图选择
    # ─────────────────────────────────────────────────────────
    def _choose_style(self):
        if self._active_region is None:
            messagebox.showwarning("提示", "请先选择或添加一个区域！", parent=self)
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="选择风格图",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self._active_region.style_pil = Image.open(path).convert("RGB")
            self._active_region.style_name = os.path.basename(path)
            self._style_lbl.config(
                text=self._active_region.style_name, fg=GREEN)
            self._update_style_thumb(self._active_region.style_pil)
            self._refresh_listbox()
            self._status(
                f"{self._active_region.name} → 风格图：{self._active_region.style_name}")
        except Exception as e:
            messagebox.showerror("加载失败", str(e), parent=self)

    def _update_style_thumb(self, pil_img: Image.Image):
        """更新风格图缩略图显示。"""
        thumb = pil_img.copy()
        thumb.thumbnail((120, 80), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        self._style_thumb_lbl.config(image=photo)
        self._style_thumb_lbl._photo = photo  # 防 GC

    # ─────────────────────────────────────────────────────────
    # 画笔绘制
    # ─────────────────────────────────────────────────────────
    def _canvas_to_image(self, cx: int, cy: int):
        """将画布坐标转换为内容图坐标。"""
        ix = int(cx / self._scale_x)
        iy = int(cy / self._scale_y)
        return ix, iy

    def _on_paint_start(self, event):
        if self._content_pil is None or self._active_region is None:
            return
        self._drawing = True
        self._last_xy = (event.x, event.y)
        self._paint_at(event.x, event.y)

    def _on_paint_move(self, event):
        if not self._drawing or self._active_region is None:
            return
        # 连线插值：避免快速移动留下空洞
        if self._last_xy:
            x0, y0 = self._last_xy
            x1, y1 = event.x, event.y
            dist = max(abs(x1 - x0), abs(y1 - y0))
            steps = max(1, dist // 4)
            for i in range(1, steps + 1):
                xi = int(x0 + (x1 - x0) * i / steps)
                yi = int(y0 + (y1 - y0) * i / steps)
                self._paint_at(xi, yi)
        self._last_xy = (event.x, event.y)
        self._refresh_canvas()

    def _on_paint_end(self, event):
        self._drawing = False
        self._last_xy = None
        self._refresh_canvas()

    def _paint_at(self, cx: int, cy: int):
        """在指定画布坐标处绘制/擦除笔触。"""
        ix, iy = self._canvas_to_image(cx, cy)
        r = int(self._brush_size.get() / self._scale_x)
        if self._erase_mode.get():
            self._active_region.erase_circle(ix, iy, r)
        else:
            self._active_region.draw_circle(ix, iy, r)
        # 同步到 alpha 值
        self._active_region.alpha = self._alpha_var.get()
        # 更新叠加层
        self._rebuild_overlay()

    def _rebuild_overlay(self):
        """重新生成所有区域掩码的彩色叠加层。"""
        if self._content_pil is None:
            return
        W, H = self._content_pil.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for region in self._regions:
            mask = region.get_mask()
            if mask is None:
                continue
            # 将掩码转为 RGBA，使用区域颜色
            hex_color = region.color.lstrip("#")
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # 绘制半透明彩色叠加
            color_layer = Image.new("RGBA", (W, H), rgb + (120,))
            overlay = Image.composite(
                color_layer, overlay,
                mask.point(lambda p: p)  # 用掩码控制合成
            )
        self._overlay = overlay

    def _clear_current_mask(self):
        if self._active_region and self._content_pil:
            W, H = self._content_pil.size
            self._active_region.init_mask(W, H)
            self._rebuild_overlay()
            self._refresh_canvas()

    # ─────────────────────────────────────────────────────────
    # 画布刷新
    # ─────────────────────────────────────────────────────────
    def _refresh_canvas(self):
        """将内容图 + 掩码叠加层绘制到画布。"""
        if self._content_pil is None:
            return

        cw = self._canvas.winfo_width() or CANVAS_W
        ch = self._canvas.winfo_height() or CANVAS_H
        W, H = self._content_pil.size

        # 计算显示比例
        scale = min(cw / W, ch / H)
        dw = int(W * scale)
        dh = int(H * scale)
        self._scale_x = dw / W
        self._scale_y = dh / H

        # 合成：内容图 + 叠加层
        display = self._content_pil.copy().convert("RGBA")
        if self._overlay:
            overlay_resized = self._overlay.resize((W, H), Image.LANCZOS)
            display = Image.alpha_composite(display, overlay_resized)

        display = display.convert("RGB").resize((dw, dh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        self._photo_content = photo

        self._canvas.delete("all")
        self._canvas.create_image(
            cw // 2, ch // 2, anchor="center", image=photo)

        # 绘制当前画笔预览圆（如果有活动区域）
        if self._active_region:
            color = self._active_region.color
            r = self._brush_size.get() // 2
            # 在画布中心位置显示当前笔刷大小示意
            self._canvas.create_oval(
                8, 8, 8 + r * 2, 8 + r * 2,
                outline=color, width=2
            )
            mode_text = "擦除" if self._erase_mode.get() else "绘制"
            self._canvas.create_text(
                14 + r * 2, 18, anchor="w",
                text=f"{self._active_region.name} | {mode_text}",
                fill=color, font=("微软雅黑", 8)
            )

    # ─────────────────────────────────────────────────────────
    # 执行风格迁移
    # ─────────────────────────────────────────────────────────
    def _run(self):
        if self._content_pil is None:
            messagebox.showwarning("提示", "请先加载内容图！", parent=self)
            return
        if not self._regions:
            messagebox.showwarning("提示", "请至少添加一个区域！", parent=self)
            return

        # 检查所有区域都有风格图和掩码
        valid_regions = []
        for r in self._regions:
            if r.style_pil is None:
                messagebox.showwarning(
                    "提示", f"{r.name} 未选择风格图，该区域将被跳过。",
                    parent=self)
                continue
            if not r.has_mask():
                messagebox.showwarning(
                    "提示", f"{r.name} 掩码为空，该区域将被跳过。",
                    parent=self)
                continue
            valid_regions.append(r)

        if not valid_regions:
            messagebox.showwarning(
                "提示", "没有有效区域（需要有风格图且掩码不为空）！",
                parent=self)
            return

        self._btn_run.config(state="disabled", text="处理中...")
        self._status("正在执行掩码风格迁移，请稍候...")
        threading.Thread(
            target=self._run_worker,
            args=(valid_regions,),
            daemon=True
        ).start()

    def _run_worker(self, valid_regions):
        try:
            # 构建 regions 参数
            regions_data = [
                {
                    'mask':  r.get_mask(),
                    'style': r.style_pil,
                    'alpha': r.alpha,
                }
                for r in valid_regions
            ]
            result, ms = self._st.masked_transfer(
                self._content_pil, regions_data
            )
            self._result_pil = result
            self.after(0, self._show_result)
            self._status(
                f"完成！推理耗时 {ms:.1f} ms  ({len(valid_regions)} 个区域)"
            )
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "推理错误", str(e), parent=self))
            self._status("错误：" + str(e))
        finally:
            self.after(0, lambda: self._btn_run.config(
                state="normal", text="▶ 执行掩码风格迁移"
            ))

    def _show_result(self):
        if self._result_pil is None:
            return
        cw = self._result_canvas.winfo_width() or CANVAS_W
        ch = self._result_canvas.winfo_height() or 200
        thumb = self._result_pil.copy()
        thumb.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        self._photo_result = photo
        self._result_canvas.delete("all")
        self._result_canvas.create_image(
            cw // 2, ch // 2, anchor="center", image=photo)

    # ─────────────────────────────────────────────────────────
    # 保存
    # ─────────────────────────────────────────────────────────
    def _save_result(self):
        if self._result_pil is None:
            messagebox.showwarning("提示", "请先执行风格迁移！", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("所有文件", "*.*")]
        )
        if path:
            self._result_pil.save(path)
            self._status("结果已保存：" + os.path.basename(path))

    # ─────────────────────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────────────────────
    def _status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))
