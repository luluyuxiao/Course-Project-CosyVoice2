import sys
import os
import torchaudio
from modelscope import snapshot_download

# 添加依赖路径
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.')

from cosyvoice.cli.cosyvoice import CosyVoice2


def auto_fix_model_config():
    """
    由于自己下载过程中官网出现了部分配置上的bug，因此可以加上这段进行判断，判断模型配置文件是否为空
    自动下载模型，并修复官方 yaml 配置文件中 qwen_pretrain_path 为空的问题。
    确保在任何上都能一键运行，无需手动修改配置文件。
    """
    print("正在检查/下载模型权重 (ModelScope)...")
    try:
        # 1. 获取模型下载后的绝对路径
        model_dir = snapshot_download('iic/CosyVoice2-0.5B')
        yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')

        # 兼容性处理：有时候下载下来叫 cosyvoice2.yaml
        if not os.path.exists(yaml_path):
            yaml_path_v2 = os.path.join(model_dir, 'cosyvoice2.yaml')
            if os.path.exists(yaml_path_v2):
                yaml_path = yaml_path_v2

        print(f"模型配置文件路径: {yaml_path}")

        # 2. 读取配置文件
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 3. 动态计算 LLM 路径 (CosyVoice-BlankEN 文件夹)
        llm_path = os.path.join(model_dir, 'CosyVoice-BlankEN')

        # 4. 检查并修复 Bug
        # 查找那个空的配置项: qwen_pretrain_path: ''
        if "qwen_pretrain_path: ''" in content or 'qwen_pretrain_path: ""' in content:
            print("检测到配置文件存在路径缺失 Bug，正在自动修复...")

            # 替换为正确的绝对路径
            new_content = content.replace(
                "qwen_pretrain_path: ''",
                f"qwen_pretrain_path: '{llm_path}'"
            ).replace(
                'qwen_pretrain_path: ""',
                f"qwen_pretrain_path: '{llm_path}'"
            )

            # 写回文件
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("配置文件修复完成！✅")
        else:
            print("配置文件检查通过，无需修复。")

    except Exception as e:
        print(f"自动修复过程中出现警告 (不影响后续尝试): {e}")


def main():
    # 先运行自动修复逻辑
    auto_fix_model_config()

    # 初始化模型
    print("正在初始化 CosyVoice 2 推理引擎...")
    # 这里会自动使用刚才修好的配置
    cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

    # 准备参数
    # 确保在项目根目录或者 src 目录下能找到这个文件
    # 建议提交时把 ref.wav 放在项目根目录
    ref_audio_path = 'ref.wav'

    if not os.path.exists(ref_audio_path):
        if os.path.exists(os.path.join('results', 'ref.wav')):
            ref_audio_path = os.path.join('results', 'ref.wav')
        else:
            print(f"错误：找不到参考音频 {ref_audio_path}，请确保文件存在！")
            return

    """
    此处的prompt_text最好要和ref.wav音频保持一致 可以达到最好的效果
    """
    prompt_text = "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"
    target_text = "<|zh|><|happy|>这是CosyVoice二代的测试，不仅能克隆音色，还能模仿情感，简直太强了！"

    print(f"正在生成: {target_text}")

    # 执行推理
    output = cosyvoice.inference_zero_shot(target_text, prompt_text, ref_audio_path, stream=False)

    # 保存结果到当前目录
    for i, j in enumerate(output):
        torchaudio.save(f'generated_cosyvoice_{i}.wav', j['tts_speech'], 22050)
        print(f"生成完成！已保存为 generated_cosyvoice_{i}.wav")


if __name__ == "__main__":
    main()

