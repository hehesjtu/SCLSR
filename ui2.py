import sys, os, cv2, numpy as np
from math import log10
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
    QProgressDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen
from PyQt5.QtCore import Qt, QSize, QRect, pyqtSignal

# ---------- Basic utilities ----------
def arr_to_qpixmap_letterbox(arr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    if arr is None: return QPixmap()
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4: arr = arr[:, :, :3]
    if arr.dtype != np.uint8: arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    if w == 0 or h == 0 or target_w <= 0 or target_h <= 0: return QPixmap()
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
    resized = cv2.resize(arr, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 220, dtype=np.uint8)
    y0 = (target_h - new_h) // 2; x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    qimg = QImage(canvas.data, target_w, target_h, 3*target_w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def _letterbox_params(img_w: int, img_h: int, box_w: int, box_h: int):
    scale = min(box_w / img_w, box_h / img_h)
    new_w, new_h = max(1, int(img_w*scale)), max(1, int(img_h*scale))
    x0 = (box_w - new_w) // 2; y0 = (box_h - new_h) // 2
    return scale, x0, y0, new_w, new_h

def _find_cjk_font():
    candidates = []
    win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    candidates += [os.path.join(win_fonts, x) for x in [
        "msyh.ttc","msyhbd.ttc","simhei.ttf","simsun.ttc","NotoSansCJK-Regular.ttc","SourceHanSansCN-Regular.otf"
    ]]
    candidates += ["/System/Library/Fonts/PingFang.ttc","/Library/Fonts/PingFang.ttc"]
    for p in candidates:
        if os.path.isfile(p): return p
    return None

def draw_placeholder(text=''):
    from PIL import Image, ImageDraw, ImageFont
    w, h = 640, 360
    img = Image.new("RGB", (w, h), (220, 220, 220)); draw = ImageDraw.Draw(img)
    font_path = _find_cjk_font()
    if font_path is None:
        text = 'Click "Load Image"'
        try: font = ImageFont.truetype("arial.ttf", 18)
        except Exception: font = ImageFont.load_default()
    else:
        try: font = ImageFont.truetype(font_path, 22)
        except Exception:
            text = 'Click "Load Image"'
            try: font = ImageFont.truetype("arial.ttf", 18)
            except Exception: font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0,0), text, font=font); text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)
    draw.text(((w-text_w)//2, (h-text_h)//2), text, fill=(0,0,0), font=font)
    return np.array(img)

# ---------- Metrics (luminance channel) ----------
def _to_y_255(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[...,0].astype(np.float32); g = rgb_u8[...,1].astype(np.float32); b = rgb_u8[...,2].astype(np.float32)
    return 0.257*r + 0.504*g + 0.098*b + 16.0

def psnr_imgA_imgB(imgA_rgb: np.ndarray, imgB_rgb: np.ndarray) -> float:
    assert imgA_rgb.shape == imgB_rgb.shape
    yA, yB = _to_y_255(imgA_rgb), _to_y_255(imgB_rgb)
    mse = float(np.mean((yA - yB)**2))
    return 99.0 if mse <= 1e-12 else 10.0 * log10((255.0**2)/mse)

def _gaussian_window(ks=11, sigma=1.5):
    ax = np.arange(ks) - ks//2; g = np.exp(-0.5*(ax/sigma)**2)
    w = np.outer(g,g).astype(np.float32); w /= w.sum(); return w

def ssim_imgA_imgB(imgA_rgb: np.ndarray, imgB_rgb: np.ndarray, K1=0.01, K2=0.03, L=255.0):
    assert imgA_rgb.shape == imgB_rgb.shape
    y1 = _to_y_255(imgA_rgb).astype(np.float32); y2 = _to_y_255(imgB_rgb).astype(np.float32)
    W = _gaussian_window(11, 1.5)
    mu1 = cv2.filter2D(y1,-1,W); mu2 = cv2.filter2D(y2,-1,W)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.filter2D(y1*y1,-1,W) - mu1_sq
    sigma2_sq = cv2.filter2D(y2*y2,-1,W) - mu2_sq
    sigma12   = cv2.filter2D(y1*y2,-1,W) - mu12
    C1, C2 = (K1*L)**2, (K2*L)**2
    ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(ssim_map.mean())

# ---------- Label with selectable zoom region ----------
class ZoomableLabel(QLabel):
    roiFinished = pyqtSignal(QRect)
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.setMouseTracking(True)
        self._zoom_mode=False; self._dragging=False; self._p0=None; self._p1=None
    def setZoomMode(self, enabled: bool):
        self._zoom_mode = enabled; self._dragging=False; self._p0=self._p1=None; self.update()
    def paintEvent(self, e):
        super().paintEvent(e)
        if self._zoom_mode and self._dragging and self._p0 and self._p1:
            qp = QPainter(self); pen = QPen(Qt.red); pen.setWidth(2); qp.setPen(pen)
            qp.drawRect(QRect(self._p0, self._p1).normalized())
    def mousePressEvent(self, e):
        if self._zoom_mode and e.button()==Qt.LeftButton:
            self._dragging=True; self._p0=e.pos(); self._p1=e.pos(); self.update()
        else: super().mousePressEvent(e)
    def mouseMoveEvent(self, e):
        if self._zoom_mode and self._dragging: self._p1=e.pos(); self.update()
        else: super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        if self._zoom_mode and self._dragging and e.button()==Qt.LeftButton:
            self._dragging=False; self._p1=e.pos(); r = QRect(self._p0, self._p1).normalized()
            if r.width()>5 and r.height()>5: self.roiFinished.emit(r)
            self.update()
        else: super().mouseReleaseEvent(e)

# ---------- Main window ----------
class DetailEnhancementApp(QWidget):
    def __init__(self, enhance_fn):
        super().__init__()
        self.enhance_fn = enhance_fn
        self.setWindowTitle("基于自校正学习的遥感图像超分辨重建软件")
        self.resize(1280, 780)

        # Three display-image caches and the images used for metrics.
        self.originalImage = None      # Nearest-neighbor upsampled image.
        self.enhancedImage = None      # Model-upsampled image.
        self.residualImage = None      # |nearest-neighbor xS - model xS|.
        self._metric_input = None
        self._metric_enhanced = None

        # Toolbar buttons.
        self.btnLoad = QPushButton("载入图像")
        self.btnSave = QPushButton("保存超分辨率图像")
        self.btnPSNR = QPushButton("计算 PSNR")
        self.btnSSIM = QPushButton("计算 SSIM")
        self.btnResidual = QPushButton("生成高频层图像")
        self.btnZoom = QPushButton("局部放大")
        self.btnRestore = QPushButton("恢复原图")
        for b in (self.btnSave, self.btnPSNR, self.btnSSIM, self.btnResidual, self.btnZoom, self.btnRestore):
            b.setEnabled(False)

        self.btnLoad.clicked.connect(self.loadImage)
        self.btnSave.clicked.connect(self.saveEnhancedImage)
        self.btnPSNR.clicked.connect(self.computePSNR)
        self.btnSSIM.clicked.connect(self.computeSSIM)
        self.btnResidual.clicked.connect(self.generateResidual)
        self.btnZoom.clicked.connect(self.startZoom)
        self.btnRestore.clicked.connect(self.restoreFullView)

        # Three image display panels.
        ph = draw_placeholder()
        self.lblOriginalImage = ZoomableLabel(alignment=Qt.AlignCenter)
        self.lblEnhancedImage = ZoomableLabel(alignment=Qt.AlignCenter)
        self.lblResidualImage = QLabel(alignment=Qt.AlignCenter)
        for lbl, size in [(self.lblOriginalImage, QSize(640,360)),
                          (self.lblEnhancedImage, QSize(640,360)),
                          (self.lblResidualImage, QSize(480,320))]:
            lbl.setMinimumSize(size)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setPixmap(arr_to_qpixmap_letterbox(ph, size.width(), size.height()))
        self.lblOriginalImage.roiFinished.connect(self._on_roi_finished_from_original)
        self.lblEnhancedImage.roiFinished.connect(self._on_roi_finished_from_enhanced)

        self.captionOriginal = QLabel("原始图像", alignment=Qt.AlignCenter)
        self.captionEnhanced = QLabel("超分辨率图像", alignment=Qt.AlignCenter)
        self.captionResidual = QLabel("高频层", alignment=Qt.AlignCenter)

        self.lblPSNR = QLabel("PSNR: "); self.lblSSIM = QLabel("SSIM: ")
        for lab in (self.lblPSNR, self.lblSSIM): lab.setFont(QFont("Microsoft YaHei", 12))

        # Copyright notice at the top right; do not display the scale factor.
        self.lblCopyright = QLabel("Copyright@中国矿业大学智能检测和模式识别研究所",
                                   alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.lblCopyright.setFont(QFont("Microsoft YaHei", 9))
        self.lblCopyright.setStyleSheet("color: gray;")

        # Layout.
        top = QHBoxLayout(); btns = QHBoxLayout()
        for w in (self.btnLoad, self.btnSave, self.btnPSNR, self.btnSSIM, self.btnResidual, self.btnZoom, self.btnRestore):
            btns.addWidget(w)
        top.addLayout(btns); top.addWidget(QLabel(""), stretch=1); top.addWidget(self.lblCopyright)

        grid = QGridLayout(); grid.addWidget(self.lblOriginalImage, 0, 0); grid.addWidget(self.lblEnhancedImage, 0, 1)
        caps = QHBoxLayout(); caps.addWidget(self.captionOriginal); caps.addWidget(self.captionEnhanced)

        bottom = QHBoxLayout()
        metrics = QVBoxLayout(); metrics.addWidget(self.lblPSNR); metrics.addWidget(self.lblSSIM)
        resid = QVBoxLayout(); resid.addWidget(self.lblResidualImage); resid.addWidget(self.captionResidual)
        bottom.addLayout(metrics, 12); bottom.addLayout(resid, 8)

        main = QVBoxLayout(); main.setContentsMargins(10,16,10,8)
        main.addLayout(top); main.addLayout(grid); main.addLayout(caps); main.addLayout(bottom)
        self.setLayout(main)

        # Zoom state.
        self._zoom_enabled = False
        self._backup_full_original = None
        self._backup_full_enhanced = None

    # ---------- Basic display ----------
    def _set_image_on_label(self, label: QLabel, arr: np.ndarray):
        w, h = label.width(), label.height()
        label.setPixmap(arr_to_qpixmap_letterbox(arr, w, h))

    # ---------- Map label coordinates to image coordinates ----------
    def _label_rect_to_image_rect(self, label: QLabel, img: np.ndarray, rect: QRect):
        box_w, box_h = label.width(), label.height()
        img_h, img_w = img.shape[:2]
        scale, x0, y0, new_w, new_h = _letterbox_params(img_w, img_h, box_w, box_h)
        x1 = max(rect.left(), x0);  y1 = max(rect.top(),  y0)
        x2 = min(rect.right(), x0 + new_w - 1); y2 = min(rect.bottom(), y0 + new_h - 1)
        if x2 <= x1 or y2 <= y1: return None
        img_x1 = int((x1 - x0) / scale); img_y1 = int((y1 - y0) / scale)
        img_x2 = int((x2 - x0) / scale); img_y2 = int((y2 - y0) / scale)
        img_x1 = max(0, min(img_x1, img_w - 2)); img_y1 = max(0, min(img_y1, img_h - 2))
        img_x2 = max(1, min(img_x2, img_w - 1)); img_y2 = max(1, min(img_y2, img_h - 1))
        return (img_x1, img_y1, img_x2, img_y2)

    def _apply_zoom_with_roi(self, roi_img_rect):
        if self.originalImage is None or self.enhancedImage is None: return
        x1, y1, x2, y2 = roi_img_rect
        crop_o = self.originalImage[y1:y2, x1:x2].copy()
        crop_e = self.enhancedImage[y1:y2, x1:x2].copy()
        if self._backup_full_original is None: self._backup_full_original = self.originalImage
        if self._backup_full_enhanced is None: self._backup_full_enhanced = self.enhancedImage
        self._set_image_on_label(self.lblOriginalImage, crop_o)
        self._set_image_on_label(self.lblEnhancedImage, crop_e)

    # ---------- Zoom controls ----------
    def startZoom(self):
        if self.originalImage is None or self.enhancedImage is None:
            QMessageBox.information(self, "提示", "请先载入图像"); return
        self._zoom_enabled = True
        self.lblOriginalImage.setZoomMode(True)
        self.lblEnhancedImage.setZoomMode(True)

    def restoreFullView(self):
        self._zoom_enabled = False
        self.lblOriginalImage.setZoomMode(False)
        self.lblEnhancedImage.setZoomMode(False)
        self._backup_full_original = None; self._backup_full_enhanced = None
        if self.originalImage is not None: self._set_image_on_label(self.lblOriginalImage, self.originalImage)
        if self.enhancedImage is not None: self._set_image_on_label(self.lblEnhancedImage, self.enhancedImage)

    def _on_roi_finished_from_original(self, rect: QRect):
        if not self._zoom_enabled or self.originalImage is None: return
        roi = self._label_rect_to_image_rect(self.lblOriginalImage, self.originalImage, rect)
        if roi is None: return
        self._apply_zoom_with_roi(roi)

    def _on_roi_finished_from_enhanced(self, rect: QRect):
        if not self._zoom_enabled or self.enhancedImage is None: return
        roi = self._label_rect_to_image_rect(self.lblEnhancedImage, self.enhancedImage, rect)
        if roi is None: return
        self._apply_zoom_with_roi(roi)

    # ---------- Load, save, metrics, and high-frequency residual ----------
    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.jpg *.png *.bmp *.tif)")
        if not path: return

        progress = QProgressDialog(self)
        progress.setRange(0, 100); progress.setMinimumDuration(0)
        progress.setFixedWidth(600); progress.setWindowModality(Qt.WindowModal)
        progress.setLabelText("正在进行图像超分辨率…"); progress.setWindowTitle("遥感图像超分辨率")
        progress.setWindowFlag(Qt.WindowContextHelpButtonHint, False); progress.show()

        try:
            bgr = cv2.imread(path); assert bgr is not None, 'OpenCV 无法读取该图像'
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Match the run_*_app interface, which returns eval_sr and dbg.
            eval_sr, dbg = self.enhance_fn(img_rgb, progress, return_debug=True)

            self.originalImage = dbg["for_original_window"]
            self.enhancedImage = dbg["sr_from_input_x4"]
            self.residualImage = dbg["highfreq_abs"]

            self._metric_input = dbg["metric_ref_input"]
            self._metric_enhanced = dbg["metric_pred_enhanced"]

            for b in (self.btnSave, self.btnPSNR, self.btnSSIM, self.btnResidual, self.btnZoom, self.btnRestore):
                b.setEnabled(True)
            self.lblPSNR.setText("PSNR: "); self.lblSSIM.setText("SSIM: ")

            self._set_image_on_label(self.lblOriginalImage, self.originalImage)
            self._set_image_on_label(self.lblEnhancedImage, self.enhancedImage)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
        finally:
            progress.close()

    def saveEnhancedImage(self):
        if self.enhancedImage is None:
            QMessageBox.warning(self, "提示", "请先载入图像"); return
        path, _ = QFileDialog.getSaveFileName(self, "保存超分辨率图像", "", "PNG 图像 (*.png)")
        if not path: return
        cv2.imwrite(path, cv2.cvtColor(self.enhancedImage, cv2.COLOR_RGB2BGR))
        QMessageBox.information(self, "完成", "已保存超分辨率图像")

    def computePSNR(self):
        if self._metric_input is None or self._metric_enhanced is None:
            QMessageBox.warning(self, "提示", "请先载入图像"); return
        v = psnr_imgA_imgB(self._metric_input, self._metric_enhanced)
        self.lblPSNR.setText(f"PSNR: {v:.2f} dB")

    def computeSSIM(self):
        if self._metric_input is None or self._metric_enhanced is None:
            QMessageBox.warning(self, "提示", "请先载入图像"); return
        v = ssim_imgA_imgB(self._metric_input, self._metric_enhanced)
        self.lblSSIM.setText(f"SSIM: {v:.4f}")

    def generateResidual(self):
        if self.residualImage is None:
            QMessageBox.warning(self, "提示", "请先载入图像"); return
        self._set_image_on_label(self.lblResidualImage, self.residualImage)

    # ---------- Responsive redraw ----------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.originalImage is not None:
            self._set_image_on_label(self.lblOriginalImage, self.originalImage)
        if self.enhancedImage is not None:
            self._set_image_on_label(self.lblEnhancedImage, self.enhancedImage)
        if self.residualImage is not None:
            self._set_image_on_label(self.lblResidualImage, self.residualImage)

# ---------- Entry point ----------
def run_app(enhance_fn):
    app = QApplication(sys.argv); app.setFont(QFont("Microsoft YaHei", 11))
    win = DetailEnhancementApp(enhance_fn); win.show()
    sys.exit(app.exec_())
