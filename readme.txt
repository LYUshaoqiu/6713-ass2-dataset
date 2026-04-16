
HateXplain和tweeteval的数据集维度已经被统一了
hate/ofensive/normal，对应的值分别为2，1，0
CsV的第一列是评论内容 ，第二列是维度打分。

model 具体参数的下载链接（大约1GB），下载后将里面两个文件夹saved_model_roberta_hx和saved_model_roberta_te放入和predict.py相同的文件夹内
链接： https://drive.google.com/drive/folders/1sTdeJnDs5u6KIKBRIk1DSLhYbojrhvRD?usp=sharing
使用方法：
单条文本预测：
python predict.py --text "I hate all minorities"
批量文件预测（每行一句）：
python predict.py --file my_texts.txt
指定模型（HateXplain 或 TweetEval 训练的）：
python predict.py --text "some text" --model saved_model_roberta_hx
python predict.py --text "some text" --model saved_model_roberta_te

输出示例：
Text:       I hate all minorities
Prediction: Hate  (confidence: 0.9231)

加 --verbose 可看每个类别的概率：
Text:       I hate all minorities
Prediction: Hate  (confidence: 0.9231)
  Non-hate  : 0.0769
  Hate      : 0.9231


注意：如果步下载模型需要先运行 step3_twitter_roberta_finetune.py 训练完成并保存模型到 saved_model_roberta_hx/ 或 saved_model_roberta_te/ 后，predict.py 才能正常使用。



自建数据集已经准备并且分类好了
