"""
智能去水印 Web 应用 - Flask 后端
使用 OpenCV inpaint 算法 (INPAINT_TELEA) 去除图片水印
"""
import base64
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 限制上传图片最大 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    将前端传来的 Base64 字符串解码为 OpenCV 图像矩阵 (BGR)
    自动处理 data:image/...;base64, 前缀
    """
    # 移除可能存在的 data URL 前缀
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]

    # 解码 Base64 → 字节流 → numpy 数组 → OpenCV 图像
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("无法解码图片，请确保上传的是有效图片")

    return img


def encode_image_to_base64(img: np.ndarray) -> str:
    """
    将 OpenCV 图像矩阵编码为 Base64 字符串（含 data URL 前缀）
    """
    # 编码为 PNG 格式的字节流
    success, buffer = cv2.imencode('.png', img)
    if not success:
        raise ValueError("图片编码失败")

    # 转为 Base64 并添加 data URL 前缀
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def detect_watermark_pixels(roi: np.ndarray) -> np.ndarray:
    """
    智能检测 ROI 区域内哪些像素是水印（而非原始背景）

    核心思路：
      - 水印通常以半透明叠加或清晰边缘的形式存在
      - 通过「边缘检测 + 局部亮度偏差 + 色彩偏差」三重分析来定位水印像素
      - 只标记水印像素，保留背景像素不动 → 避免「橡皮擦」效果

    参数:
        roi: 框选区域图像 (BGR)

    返回:
        二值掩码 (0=背景保留, 255=水印需修复)，尺寸与 roi 相同
    """
    roi_h, roi_w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ============ 方法一：边缘检测（捕获水印文字/Logo 轮廓） ============
    # 低阈值+高阈值，捕获水印的半透明边缘
    edges = cv2.Canny(gray, 25, 90)
    # 膨胀边缘，让水印笔画内部也被覆盖
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)

    # ============ 方法二：局部亮度偏差（捕获半透明水印） ============
    # 大核高斯模糊模拟「没有水印的局部背景亮度」
    ks = max(5, min(roi_w, roi_h) // 8)
    if ks % 2 == 0:
        ks += 1  # 必须为奇数
    bg_luminance = cv2.GaussianBlur(gray, (ks, ks), 0)
    # 含水印的像素与局部背景存在亮度差
    luma_diff = cv2.absdiff(gray, bg_luminance)
    _, bright_mask = cv2.threshold(luma_diff, 10, 255, cv2.THRESH_BINARY)

    # ============ 方法三：色彩偏差（捕获彩色水印） ============
    # 将 ROI 转到 Lab 色彩空间，A/B 通道对色彩敏感
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a_channel = roi_lab[:, :, 1]  # 绿←→红
    b_channel = roi_lab[:, :, 2]  # 蓝←→黄
    # 对 A/B 通道做同样的局部偏差检测
    ks_color = max(5, min(roi_w, roi_h) // 10)
    if ks_color % 2 == 0:
        ks_color += 1
    bg_a = cv2.GaussianBlur(a_channel, (ks_color, ks_color), 0)
    bg_b = cv2.GaussianBlur(b_channel, (ks_color, ks_color), 0)
    color_diff = cv2.absdiff(a_channel, bg_a) + cv2.absdiff(b_channel, bg_b)
    _, color_mask = cv2.threshold(color_diff, 15, 255, cv2.THRESH_BINARY)

    # ============ 方法四：自适应阈值（捕获高对比度水印文字） ============
    # 对低对比度水印效果好
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6
    )
    # 同时做一次正向阈值，合并
    adaptive2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 6
    )
    adaptive_mask = cv2.bitwise_or(adaptive, adaptive2)

    # ============ 合并所有检测结果 ============
    combined = cv2.bitwise_or(edge_mask, bright_mask)
    combined = cv2.bitwise_or(combined, color_mask)
    combined = cv2.bitwise_or(combined, adaptive_mask)

    # ============ 形态学后处理 ============
    kernel = np.ones((3, 3), np.uint8)
    # 闭运算：填充水印内部小空洞
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 开运算：去除孤立的噪点
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # ============ 安全检查 ============
    # 如果检测到的像素极少（<3%），说明此区域可能没有明显水印特征
    # → 回退到整框修复，保证用户至少能去除
    watermark_ratio = np.sum(combined > 0) / (roi_w * roi_h)
    if watermark_ratio < 0.03:
        return np.ones((roi_h, roi_w), dtype=np.uint8) * 255

    return combined


def remove_watermark(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    智能去水印：先检测水印像素，再精准修复

    与粗暴填充整块矩形不同，本函数会：
      1. 分析矩形区域内的像素，判断哪些是水印、哪些是背景
      2. 只对水印像素使用 cv2.inpaint 修复
      3. 保留框内与背景一致的像素不动

    参数:
        img: 原始图像 (BGR)
        x, y: 水印矩形左上角坐标
        w, h: 水印矩形的宽度和高度

    返回:
        处理后图像 (BGR)
    """
    img_height, img_width = img.shape[:2]

    # 边界裁剪：确保坐标不超出图像范围
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))

    # 提取 ROI 并智能检测水印像素
    roi = img[y:y + h, x:x + w]

    # 创建智能掩码：只标记水印像素，不碰背景
    roi_mask = detect_watermark_pixels(roi)

    # 将 ROI 掩码嵌入全图尺寸的掩码中
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    full_mask[y:y + h, x:x + w] = roi_mask

    # 使用 INPAINT_TELEA 算法精准修复水印像素
    # inpaintRadius=5：略大半径让修复过渡更自然
    result = cv2.inpaint(img, full_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # ============ 边缘羽化：让修复区域与周围自然融合 ============
    # 对修复区域的边界做轻度高斯模糊，消除 inpaint 可能产生的硬边
    if np.sum(roi_mask > 0) > 0:
        # 创建软边界掩码（只羽化水印区域的边缘）
        roi_mask_float = roi_mask.astype(np.float32) / 255.0
        # 膨胀后减去原掩码 → 得到边界带
        dilated = cv2.dilate(roi_mask, np.ones((5, 5), np.uint8), iterations=1)
        boundary = cv2.subtract(dilated, roi_mask)
        # 对边界做高斯模糊作为混合权重
        boundary_blur = cv2.GaussianBlur(boundary.astype(np.float32), (5, 5), 0) / 255.0

        if np.sum(boundary_blur) > 0:
            # 在 ROI 区域内按权重混合原图和修复结果
            result_roi = result[y:y + h, x:x + w].astype(np.float32)
            orig_roi = img[y:y + h, x:x + w].astype(np.float32)

            # 边界区域：加权混合 | 非边界区域：直接用修复结果
            # boundary_blur 只在边界处有值
            blended = result_roi.copy()
            for c in range(3):
                blended[:, :, c] = (
                    result_roi[:, :, c] * (1 - boundary_blur * 0.5) +
                    orig_roi[:, :, c] * (boundary_blur * 0.5)
                )

            result[y:y + h, x:x + w] = blended.astype(np.uint8)

    return result


# ==================== 路由 ====================

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')


@app.route('/api/remove-watermark', methods=['POST'])
def api_remove_watermark():
    """
    去水印 API

    请求 JSON:
        image: 图片的 Base64 编码字符串
        x, y, w, h: 水印矩形区域坐标

    响应 JSON:
        success: bool
        image: 处理后的 Base64 图片 (成功时)
        error: 错误信息 (失败时)
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'}), 400

        # 获取参数
        image_b64 = data.get('image', '')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        w = int(data.get('w', 0))
        h = int(data.get('h', 0))

        # 参数校验
        if not image_b64:
            return jsonify({'success': False, 'error': '未收到图片数据'}), 400

        if w <= 0 or h <= 0:
            return jsonify({'success': False, 'error': '水印区域无效，请重新框选'}), 400

        # 解码图片
        img = decode_base64_image(image_b64)

        # 去除水印
        result_img = remove_watermark(img, x, y, w, h)

        # 编码返回
        result_b64 = encode_image_to_base64(result_img)

        return jsonify({
            'success': True,
            'image': result_b64
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器内部错误: {str(e)}'}), 500


if __name__ == '__main__':
    # debug=True 开发模式，生产环境请关闭
    app.run(host='0.0.0.0', port=5000, debug=True)
