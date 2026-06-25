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


def remove_watermark(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    使用 OpenCV INPAINT_TELEA 算法去除水印区域

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

    # 创建修复遮罩 (mask)：在水印区域标记为白色(255)，其余为黑色(0)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255

    # 使用 INPAINT_TELEA 算法进行修复
    # radius=3：修复半径，控制修复时参考周围像素的范围
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

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
