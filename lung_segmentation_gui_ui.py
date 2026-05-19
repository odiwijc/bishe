import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as nnfun
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import warnings
warnings.filterwarnings('ignore')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class STN(nn.Module):
    def __init__(self, in_channels=1, output_size=(256, 256)):
        super(STN, self).__init__()
        self.output_size = output_size
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        bs = x.size(0)
        loc_feat = self.localization(x).view(bs, -1)
        theta = self.fc_loc(loc_feat)
        theta = theta.view(bs, 2, 3)
        grid = nnfun.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = nnfun.grid_sample(x, grid, align_corners=True)
        return x_transformed


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, use_stn=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_stn = use_stn

        if self.use_stn:
            self.stn = STN(in_channels=n_channels)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ImageSegmentationDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        original_image = np.array(image)
        image = self.transform(image)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)
        return image, original_image, os.path.basename(img_path)


class SegmentationThread(QThread):
    progress_updated = pyqtSignal(int, int, str)
    segmentation_complete = pyqtSignal(list, list, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths, model, device, image_size=256):
        super().__init__()
        self.image_paths = image_paths
        self.model = model
        self.device = device
        self.image_size = image_size

    def run(self):
        try:
            dataset = ImageSegmentationDataset(self.image_paths, (self.image_size, self.image_size))
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            original_images = []
            segmentation_masks = []
            overlay_images = []
            
            self.model.eval()
            
            for i, (images, original_imgs, filenames) in enumerate(dataloader):
                images = images.to(self.device)
                with torch.no_grad():
                    outputs = self.model(images)
                    probs = torch.sigmoid(outputs)
                    masks = (probs > 0.5).float().cpu().numpy()
                
                for j in range(len(filenames)):
                    orig_img = np.array(original_imgs[j])
                    mask = masks[j][0]
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
                    )
                    mask_np = np.array(mask_img)
                    
                    overlay = np.zeros((orig_img.shape[0], orig_img.shape[1], 3), dtype=np.uint8)
                    overlay[:, :, 0] = orig_img
                    overlay[:, :, 1] = np.clip(orig_img + mask_np // 2, 0, 255)
                    overlay[:, :, 2] = mask_np
                    
                    original_images.append(orig_img)
                    segmentation_masks.append(mask_np)
                    overlay_images.append(overlay)
                
                self.progress_updated.emit(i + 1, len(dataloader), filenames[0])
            
            self.segmentation_complete.emit(original_images, segmentation_masks, overlay_images)
        except Exception as e:
            self.error_occurred.emit(str(e))


class LungSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('d:\\bishe\\lung_segmentation_gui.ui', self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.current_model_name = None
        self.current_image_index = 0
        self.images_info = []
        self.original_images = []
        self.segmentation_masks = []
        self.overlay_images = []
        
        # 编辑工具相关
        self.current_tool = 'brush'  # 'brush' or 'eraser'
        self.brush_size = 10
        self.is_drawing = False
        self.last_pos = None
        
        # 撤销/重做历史
        self.edit_history = []
        self.history_index = -1
        
        self.init_connections()
        self.load_models()

    def init_connections(self):
        self.importBtn.clicked.connect(self.import_images)
        self.segmentBtn.clicked.connect(self.start_segmentation)
        self.exportBtn.clicked.connect(self.export_results)
        self.prevBtn.clicked.connect(self.prev_image)
        self.nextBtn.clicked.connect(self.next_image)
        
        # 编辑工具连接
        self.brushBtn.clicked.connect(self.select_brush)
        self.eraserBtn.clicked.connect(self.select_eraser)
        self.brushSizeSlider.valueChanged.connect(self.update_brush_size)
        self.undoBtn.clicked.connect(self.undo)
        self.redoBtn.clicked.connect(self.redo)
        
        # 鼠标事件连接
        self.overlayLabel.mousePressEvent = self.mouse_press
        self.overlayLabel.mouseMoveEvent = self.mouse_move
        self.overlayLabel.mouseReleaseEvent = self.mouse_release

    def load_models(self):
        try:
            unet_path = r'd:\bishe\saved_models\unet_lungs_segmentation.pth'
            unet_stn_path = r'd:\bishe\saved_models\unet_stn_lungs_segmentation.pth'
            
            self.statusLabel.setText('正在加载U-Net模型...')
            QApplication.processEvents()
            
            unet = UNet(n_channels=1, n_classes=1, use_stn=False)
            checkpoint = torch.load(unet_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                unet.load_state_dict(checkpoint['model_state_dict'])
            else:
                unet.load_state_dict(checkpoint)
            unet.to(self.device)
            unet.eval()
            self.models['U-Net'] = unet
            
            self.statusLabel.setText('正在加载U-Net + STN模型...')
            QApplication.processEvents()
            
            unet_stn = UNet(n_channels=1, n_classes=1, use_stn=True)
            checkpoint = torch.load(unet_stn_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                unet_stn.load_state_dict(checkpoint['model_state_dict'])
            else:
                unet_stn.load_state_dict(checkpoint)
            unet_stn.to(self.device)
            unet_stn.eval()
            self.models['U-Net + STN'] = unet_stn

            self.current_model_name = 'U-Net'
            self.statusLabel.setText(f'模型加载完成，使用: {self.device}')
            self.infoText.append(f'模型加载成功！\n设备: {self.device}')
            self.infoText.append(f'U-Net模型: {unet_path}')
            self.infoText.append(f'U-Net+STN模型: {unet_stn_path}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'模型加载失败: {str(e)}')
            self.statusLabel.setText('模型加载失败')

    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, '选择图片', '', '图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)'
        )
        if files:
            self.images_info = files
            self.current_image_index = 0
            self.original_images = []
            self.segmentation_masks = []
            self.overlay_images = []
            self.imageCountLabel.setText(f'已导入 {len(files)} 张图片')
            self.segmentBtn.setEnabled(True)
            self.prevBtn.setEnabled(False)
            self.nextBtn.setEnabled(False)
            self.imageIndexLabel.setText('')
            self.originalLabel.setText('原图')
            self.maskLabel.setText('分割遮罩')
            self.overlayLabel.setText('叠加图')
            self.infoText.append(f'已导入 {len(files)} 张图片')

    def start_segmentation(self):
        if not self.images_info:
            QMessageBox.warning(self, '警告', '请先导入图片')
            return
        
        model_name = self.modelCombo.currentText()
        if model_name not in self.models:
            QMessageBox.warning(self, '警告', '模型未加载')
            return
        
        self.segmentBtn.setEnabled(False)
        self.importBtn.setEnabled(False)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText('正在分割...')
        
        model = self.models[model_name]
        self.segmentation_thread = SegmentationThread(
            self.images_info, model, self.device
        )
        self.segmentation_thread.progress_updated.connect(self.update_progress)
        self.segmentation_thread.segmentation_complete.connect(self.segmentation_done)
        self.segmentation_thread.error_occurred.connect(self.segmentation_error)
        self.segmentation_thread.start()

    def update_progress(self, current, total, filename):
        self.progressBar.setValue(int(current / total * 100))
        self.statusLabel.setText(f'处理中: {filename}')

    def segmentation_done(self, originals, masks, overlays):
        self.original_images = originals
        self.segmentation_masks = masks
        self.overlay_images = overlays
        self.current_image_index = 0
        
        # 重置编辑历史
        self.edit_history = []
        self.history_index = -1
        
        self.progressBar.setVisible(False)
        self.segmentBtn.setEnabled(True)
        self.importBtn.setEnabled(True)
        self.exportBtn.setEnabled(True)
        self.prevBtn.setEnabled(False)
        self.nextBtn.setEnabled(len(self.images_info) > 1)
        
        self.statusLabel.setText('分割完成')
        self.infoText.append(f'分割完成，共处理 {len(self.images_info)} 张图片')
        
        self.display_current_image()

    def segmentation_error(self, error_msg):
        self.progressBar.setVisible(False)
        self.segmentBtn.setEnabled(True)
        self.importBtn.setEnabled(True)
        QMessageBox.critical(self, '错误', f'分割过程出错: {error_msg}')
        self.statusLabel.setText('分割失败')

    def display_current_image(self):
        if not self.original_images:
            return
        
        idx = self.current_image_index
        total = len(self.images_info)
        self.imageIndexLabel.setText(f'{idx + 1} / {total}')
        
        # 显示当前图片名称
        if self.images_info:
            img_path = self.images_info[idx]
            img_name = os.path.basename(img_path)
            self.imageNameLabel.setText(f'图片名称: {img_name}')
        
        self.display_image(self.originalLabel, self.original_images[idx])
        self.display_image(self.maskLabel, self.segmentation_masks[idx])
        self.display_image(self.overlayLabel, self.overlay_images[idx], True)
        
        self.prevBtn.setEnabled(idx > 0)
        self.nextBtn.setEnabled(idx < total - 1)

    def display_image(self, label, image_array, is_color=False):
        if is_color:
            h, w, c = image_array.shape
            qimage = QImage(image_array.data, w, h, c * w, QImage.Format_RGB888)
        else:
            h, w = image_array.shape
            qimage = QImage(image_array.data, w, h, w, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            # 重置编辑历史
            self.edit_history = []
            self.history_index = -1
            self.display_current_image()

    def next_image(self):
        if self.current_image_index < len(self.images_info) - 1:
            self.current_image_index += 1
            # 重置编辑历史
            self.edit_history = []
            self.history_index = -1
            self.display_current_image()

    def export_results(self):
        if not self.segmentation_masks:
            QMessageBox.warning(self, '警告', '没有可导出的分割结果')
            return
        
        export_dir = QFileDialog.getExistingDirectory(self, '选择导出目录')
        if not export_dir:
            return
        
        try:
            mask_dir = os.path.join(export_dir, 'masks')
            overlay_dir = os.path.join(export_dir, 'overlays')
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(overlay_dir, exist_ok=True)
            
            for i, (mask, overlay, img_path) in enumerate(
                zip(self.segmentation_masks, self.overlay_images, self.images_info)
            ):
                basename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(basename)[0]
                
                mask_img = Image.fromarray(mask)
                mask_img.save(os.path.join(mask_dir, f'{name_without_ext}_mask.png'))
                
                overlay_img = Image.fromarray(overlay)
                overlay_img.save(os.path.join(overlay_dir, f'{name_without_ext}_overlay.png'))
            
            QMessageBox.information(self, '成功', f'分割结果已导出到:\n{export_dir}')
            self.infoText.append(f'结果已导出到: {export_dir}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败: {str(e)}')

    def select_brush(self):
        self.current_tool = 'brush'
        self.brushBtn.setChecked(True)
        self.eraserBtn.setChecked(False)
        self.statusLabel.setText('当前工具: 画笔')

    def select_eraser(self):
        self.current_tool = 'eraser'
        self.eraserBtn.setChecked(True)
        self.brushBtn.setChecked(False)
        self.statusLabel.setText('当前工具: 橡皮擦')

    def update_brush_size(self, value):
        self.brush_size = value
        self.brushSizeLabel.setText(f'画笔大小: {value}')

    def save_state(self):
        # 保存当前遮罩状态到历史记录
        if self.segmentation_masks:
            current_mask = self.segmentation_masks[self.current_image_index].copy()
            # 截断历史记录到当前位置
            self.edit_history = self.edit_history[:self.history_index + 1]
            self.edit_history.append(current_mask)
            self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            # 恢复到上一个状态
            self.segmentation_masks[self.current_image_index] = self.edit_history[self.history_index].copy()
            # 重新生成叠加图
            self.update_overlay()
            self.display_current_image()
            self.statusLabel.setText('已撤销')

    def redo(self):
        if self.history_index < len(self.edit_history) - 1:
            self.history_index += 1
            # 恢复到下一个状态
            self.segmentation_masks[self.current_image_index] = self.edit_history[self.history_index].copy()
            # 重新生成叠加图
            self.update_overlay()
            self.display_current_image()
            self.statusLabel.setText('已重做')

    def mouse_press(self, event):
        if not self.segmentation_masks:
            return
        
        self.is_drawing = True
        self.last_pos = event.pos()
        # 开始绘制前保存状态
        self.save_state()

    def mouse_move(self, event):
        if not self.is_drawing or not self.segmentation_masks:
            return
        
        current_pos = event.pos()
        pixmap = self.overlayLabel.pixmap()
        if pixmap:
            # 计算像素坐标
            scale_x = self.segmentation_masks[self.current_image_index].shape[1] / pixmap.width()
            scale_y = self.segmentation_masks[self.current_image_index].shape[0] / pixmap.height()
            
            x1 = int(self.last_pos.x() * scale_x)
            y1 = int(self.last_pos.y() * scale_y)
            x2 = int(current_pos.x() * scale_x)
            y2 = int(current_pos.y() * scale_y)
            
            # 绘制或擦除
            self.draw_line(x1, y1, x2, y2)
            
            # 更新叠加图
            self.update_overlay()
            self.display_current_image()
            
            self.last_pos = current_pos

    def mouse_release(self, event):
        self.is_drawing = False

    def draw_line(self, x1, y1, x2, y2):
        # 使用Bresenham算法绘制线段
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        mask = self.segmentation_masks[self.current_image_index]
        height, width = mask.shape
        
        while True:
            # 确保坐标在范围内
            if 0 <= x < width and 0 <= y < height:
                # 绘制圆形笔触
                self.draw_circle(x, y, self.brush_size)
            
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def draw_circle(self, x, y, radius):
        mask = self.segmentation_masks[self.current_image_index]
        height, width = mask.shape
        
        for i in range(max(0, x - radius), min(width, x + radius + 1)):
            for j in range(max(0, y - radius), min(height, y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    if self.current_tool == 'brush':
                        mask[j, i] = 255  # 白色
                    else:  # eraser
                        mask[j, i] = 0    # 黑色

    def update_overlay(self):
        if not self.original_images or not self.segmentation_masks:
            return
        
        orig_img = self.original_images[self.current_image_index]
        mask = self.segmentation_masks[self.current_image_index]
        
        overlay = np.zeros((orig_img.shape[0], orig_img.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = orig_img
        overlay[:, :, 1] = np.clip(orig_img + mask // 2, 0, 255)
        overlay[:, :, 2] = mask
        
        self.overlay_images[self.current_image_index] = overlay


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LungSegmentationGUI()
    window.show()
    sys.exit(app.exec_())