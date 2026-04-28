GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}
PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["ccot"] = """You are an advanced multi-modal assistant, capable of analyzing and synthesizing information from structured data tables and related visual content. Your goal is to provide accurate, context-aware, and data-driven responses based on the provided inputs.

Please follow the two steps below:

In the first step:
For the provided images and its associated question, generate a scene graph for each images includes the following:
    1. Objects that are relevant to answering the question
    2. Object attributes that are relevant to answering the question
    3. Object relationships that are relevant to answering the question
Then reason the answer of question based on scene graphs.
Put these process within <think></think>.

In the second step:
Put your your final answer in <answer></answer> based on the scene graphs.
Please try to remove irrelevant content in the final answer. Like if the question is asking for yes or no, then only answer <answer>yes</answer> after your thinking.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>Not answerable</answer> after your thinking.

Question and images are provided below. Please follow the steps as instructed.

##########
Question:
{query}
##########
"""

PROMPTS["evidence"] = """You are an advanced multi-modal assistant, capable of analyzing and synthesizing information from structured data tables and related visual content. Your goal is to provide accurate, context-aware, and data-driven responses based on the provided inputs.

Please follow the four steps below:

Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information. Then, examine each image one by one, paying special attention to aspects related to the question. Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>Not answerable</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

Question and images are provided below. Please follow the steps as instructed.

##########
Question:
{query}
##########
"""

PROMPTS["ism"] = """You are an advanced multi-modal assistant, capable of analyzing and synthesizing information from structured data tables and related visual content. Your goal is to provide accurate, context-aware, and data-driven responses based on the provided inputs.

You need to strictly follow the steps and complete them step by step.

**Step 1: Goal Alignment**
Analyze the user's question and define a concise execution strategy.
- Identify the exact anchors required (e.g., specific dates, numerical columns, row headers, or trend patterns).
- Outline the necessary operations (e.g., cross-referencing tables, calculating sums, or identifying extrema).
- **Constraints**: Use professional, data-driven language. Avoid numbered lists, restating the question, or conversational filler. Output must be a single, compact paragraph.
Wrap your plan within <plan></plan> tags.

**Step 2: Information Collection & Iterative Dynamic**
Process images in numerical order, starting strictly with <record image="1">. For each, generate a concise record in this format:
- Extract relevant data points, text, or visual information addressing the <plan> or question.
- If no relevant data is found, state "No relevant information."
- End the record with a judgment based on the following logic:
    - `Judge: Insufficient information` (if the question remains partially or fully unanswered).
    - `Judge: Sufficient information` (if the question can now be fully answered).
- If the judgment is `Insufficient information`, you must continue to process the next image.
- If you reach `Judge: Sufficient information`, you must **IMMEDIATELY STOP** and do not process any further images.
Wrap information within <record image="N"></record> tags (where N is the image index).

**Step 3: Logical Synthesis & Reasoning**
Synthesize the collected information from Step 2.
- Perform step-by-step reasoning, calculations, or comparisons as defined in your <plan>.
- Cite the specific images used (e.g., "From image 1...") in your reasoning.
- If the loop finished without sufficient information, state that the information is missing.
Wrap reasoning within <reason></reason> tags.

**Step 4: Answer Question**
Provide the final answer directly based on Step 3.
- Do NOT provide extra context or conversational filler.
- If information is missing, output <answer>Not answerable</answer>.
Wrap the final answer within <answer></answer> tags.

Question and images are provided below.
##########
Question: {query}
"""
