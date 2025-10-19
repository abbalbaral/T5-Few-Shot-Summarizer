# Dataset and Few-Shot Sampling Strategy

This project leverages the large **CNN/DailyMail dataset** (abstractive summarization variant) to train a T5-small model under few-shot conditions. The strategy was designed to test the model's ability to generalize and perform transfer learning with minimal direct supervision.

**Data Source:** The data used is the **CNN/DailyMail dataset**, an established resource for summarization research. For detailed information and licensing, refer to the Hugging Face dataset page: [CNN/DailyMail Dataset on Hugging Face](https://huggingface.co/datasets/cnn_dailymail).

---

## Training Data: K=80

The final model used for deployment was fine-tuned on a minimal subset of $K=80$ samples.

* **Subset Size:** 80 examples (Document and Summary pairs).
* **Purpose:** To simulate a real-world, data-scarce environment and demonstrate the effectiveness of using a pre-trained T5 transformer in a few-shot setting.

---

## Evaluation Data: K=1000

The evaluation and final ROUGE/BERTScore calculations were performed on a distinct, larger subset.

* **Subset Size:** 1,000 examples (Document and Summary pairs).
* **Purpose:** To ensure the performance metrics were robust and generalized across a reasonably sized, unseen evaluation set.
