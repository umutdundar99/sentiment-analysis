# Sentiment Analysis

Sentiment Analysis library that trains GPT2 and NanoGPT transformer-based models on the dataset, including text conversations about customer sentiment with the customer service agent.

:green_circle: To see explanatory data analysis, please go to [Data Analysis](#data-analysis). The necessary comments on the data are provided at the end of the notebook.
:green_circle: The test dataset evaluation is done just after the training


## Installation

Please create an environment, preferably a virtual environment.

### Create a Virtual Environment
```bash
python -m venv .venv
```
### Activate the Environment

#### Windows

```bash
source venv\Scripts\activate
```

#### Ubuntu
```bash
source venv/bin/activate
```

### Download the dependencies
```bash
pip install -e .
```

## Usage

### Data Analysis
To investigate the data distribution and characteristics, please see the [data_analysis](sentiment_analysis/data_analysis.ipynb) file and execute every cell one by one

### Pre-processing
Before the training, you must run the [process](sentiment_analysis/data/process.py) file to preprocess the data and save it to the [processed](sentiment_analysis/data/processed) folder

```bash
python3.10 sentiment_analysis/data/process.py
```

The pre-processing step includes
- Convert all text to lowercase.
- Remove speaker turns starting with "agent".
- Remove all non-alphanumeric characters and punctuation.
- Replace multiple whitespaces with a single space.
- Remove the word "customer" from the conversation.
- Filter out conversations with extremely long lengths (top 5%).
- Drop empty conversations or rows with missing values.
- Create a validation split with class balance.

### Training

GPT model can be found at [GPT](sentiment_analysis/model/nn/gpt.py). The only modification on the model is the **classifier head**.

- Please run the following script to train NanoGPT on sentiment data from scratch

```bash
sentiment_analysis train nanogpt
```

GPT-2 on sentiment data with pre-trained weights

```bash
sentiment_analysis train gpt2
```

ℹ️ Please note that the former argument is the task, the latter is the model name


## Contributing

Pull requests are not welcome :interrobang:

## License

[MIT](https://choosealicense.com/licenses/mit/)
