#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 localization_translated_v1.json 文件
移除HTML标签，只保留三个关键字段
"""

import json
import re
from html import unescape


def remove_html_tags(text):
    """移除HTML标签"""
    if not text:
        return ""

    # 先解码HTML实体
    text = unescape(text)

    # 移除HTML标签（如 <br>, <page>, <hpage> 等）
    # 将 <br> 和换行标签替换为空格
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?page>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?hpage>", " ", text, flags=re.IGNORECASE)

    # 移除其他HTML标签
    text = re.sub(r"<[^>]+>", "", text)

    # 清理多余的空格
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_json_file(input_file, output_file):
    """清理JSON文件"""
    print(f"正在读取文件: {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"原始数据: {len(data)} 条记录")

    # 清理数据
    cleaned_data = []
    skipped = 0

    for i, item in enumerate(data):
        # 提取并清理三个字段
        original_en = remove_html_tags(item.get("original_en_text", ""))
        original_zh = remove_html_tags(item.get("original_zh_text", ""))
        translated = remove_html_tags(item.get("translated_text", ""))

        # 只保留非空的记录
        if original_en and original_zh and translated:
            cleaned_data.append(
                {
                    "original_en_text": original_en,
                    "original_zh_text": original_zh,
                    "translated_text": translated,
                }
            )
        else:
            skipped += 1

        # 进度显示
        if (i + 1) % 5000 == 0:
            print(f"  处理进度: {i + 1}/{len(data)}")

    print(f"\n清理完成:")
    print(f"  有效记录: {len(cleaned_data)} 条")
    print(f"  跳过记录: {skipped} 条（空文本）")

    # 保存到新文件
    print(f"\n正在保存到: {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print("文件保存成功！")

    # 显示示例
    print("\n示例数据（第一条）:")
    print("-" * 80)
    if cleaned_data:
        example = cleaned_data[0]
        print(f"EN: {example['original_en_text'][:100]}...")
        print(f"ZH (官方): {example['original_zh_text'][:100]}...")
        print(f"ZH (重译): {example['translated_text'][:100]}...")


if __name__ == "__main__":
    clean_json_file("localization_translated_v1.json", "cleaned.json")
