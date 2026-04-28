PROMPTS = {}


PROMPTS['question_answer_examples']="""
##########
-Examples-
Question: How many % of Rep/Lean Rep people think cases have risen primally because of more testing and how many % of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy?
<plan>Locate 'Rep/Lean Rep' metrics across two thematic tables: extract the percentage for 'more testing' from the causality data and the percentage for 'federal government' from the policy responsibility data to synthesize the dual-metric response.</plan>
<record>From image 1: Table shows for Rep/Lean Rep, 62% say "More people are being tested than in previous months" as the primary reason for rise in confirmed cases.</record>
<judge>Insufficient evidence</judge>
<record>From image 2: No evidence information.</record>
<judge>Insufficient evidence</judge>
<record>From image 3: Table confirms for Rep/Lean Rep, 62% say "More people are being tested than in previous months" as the primary reason.</record>
<judge>Insufficient evidence</judge>
<record>From image 4: Table shows for Rep/Lean Rep, 30% say "The federal government" should be primarily responsible for developing and executing policies to limit the spread of COVID-19.</record>
<judge>Sufficient evidence</judge>
<reason>Based on the collected evidence: From image 1 and image 3, 62% of Rep/Lean Rep people think cases have risen primarily because of more testing. From image 4, 30% of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy. Both evidences are answered with direct data from the tables.</reason>
<answer>62% of Rep/Lean Rep people think cases have risen primarily because of more testing, and 30% of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy.</answer>
##########
Question: Of the four missions of Indian Space Program's space mission from 2012 to 2017, which mission includes the launch of least number of satellites?
<plan>Identify the four mission categories in the 2012–2017 table, count the specific satellite entries within each category, and perform a comparative analysis to isolate the category with the minimum aggregate count.</plan>
<record>From image 1: Table shows four mission categories: Earth Observation Satellites, Communication & Navigation Satellites, Space Science & Planetary Exploration Satellites, and Technology Development / Launch Vehicles. For Earth Observation Satellites, satellites listed: RISAT-1, SARAL, Resourcesat-2A, Scatsat, INSAT-3DR. Count: 5. For Communication & Navigation Satellites, satellites listed: GSAT-10, IRNSS-1A, GSAT-14, INSAT-3D, GSAT-7, IRNSS-1B, IRNSS-1C, IRNSS-1D, GSAT-16, IRNSS-1E, IRNSS-1F, IRNSS-1G, GSAT-6, GSAT-15, GSAT-9, GSAT-19E, GSAT-11, GSAT-18, GSAT-17. Count: 19. For Space Science & Planetary Exploration Satellites, satellites listed: Mars Orbiter, ASTROSAT. Count: 2. For Technology Development / Launch Vehicles, no satellites listed.</record>
<judge>Sufficient evidence</judge>
<reason>From image 1: The satellite counts are Earth Observation: 5, Communication & Navigation: 19, Space Science & Planetary Exploration: 2, and Technology Development: 0 (not a satellite-launching category). Comparing the totals, among mission categories that involve satellites, Space Science & Planetary Exploration has the fewest satellite launches with 2.</reason>
<answer>Space Science & Planetary Exploration Satellites</answer>
##########
Question: In how many hours Airbus incorporated a pop-up notification acknowledging the incident?
<plan>Scan the temporal sequence of UI mockups to pinpoint the first occurrence of the Airbus.com incident acknowledgment pop-up and extract the corresponding 'Hour' label as the definitive timestamp.</plan>
<record>From image 1: The pop-up window labeled “Airbus.com Hour 3” shows that Airbus displayed an incident acknowledgment pop-up at Hour 3.</record>
<judge>Sufficient evidence</judge>
<reason>From image 1: The pop-up acknowledging the incident appears at Hour 3, clearly shown in the bottom-right image labeled “Airbus.com Hour 3”.</reason>
<answer>3 hours</answer>
##########
Question: When did the number of tweets referencing Germanwings exceed 700,000?
<plan>Trace the 'Germanwings' tweet volume across the time-series charts to identify the specific chronological point where the quantitative value first surpasses the 700,000 threshold.</plan>
<record>From image 1: No evidence information.</record>
<judge>Insufficient evidence</judge>
<record>From image 2: No evidence information.</record>
<judge>Insufficient evidence</judge>
<record>From image 3: No evidence information.</record>
<judge>Insufficient evidence</judge>
<record>From image 4: No evidence information.</record>
<judge>Insufficient evidence</judge>
<record>From image 5: No evidence information.</record>
<judge>Insufficient evidence</judge>
<reason>No evidence was collected from the provided inputs to determine when the Germanwings tweet count exceeded 700,000.</reason>
<answer>Not answerable</answer>
##########
"""

PROMPTS['question_answer']="""You are an advanced multi-modal assistant, capable of analyzing and synthesizing information from structured data tables and related visual content. Your goal is to provide accurate, context-aware, and data-driven responses based on the provided inputs.

You need to strictly follow the steps and complete them step by step.

**Step 1: Goal Alignment**
Analyze the user's question and define a concise execution strategy.
- Identify the exact anchors required (e.g., specific dates, numerical columns, row headers, or trend patterns).
- Outline the necessary operations (e.g., cross-referencing tables, calculating sums, or identifying extrema).
- **Constraints**: Use professional, data-driven language. Avoid numbered lists, restating the question, or conversational filler. Output must be a single, compact paragraph.
Wrap your plan within <plan></plan> tags.

**Step 2: Information Collection**
Process images in numerical order, starting strictly with <record image="1">. For each, generate a record in this format:
- Only extract relevant data points, text, or visual information addressing the <plan> and question.
- If no relevant data is found, state "No relevant information."
- End the record with a judgment based on the following logic:
    - `Judge: Insufficient information` (if the question remains partially or fully unanswered).
    - `Judge: Sufficient information` (if the question can now be fully answered).
- If the judgment is `Judge: Insufficient information`, you must **CONTINUE COLLECT** information in the next image.
- If the judgment is `Judge: Sufficient information`, you must **IMMEDIATELY STOP** and do not process any further images.
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


PROMPTS["generate_step_1"] = """You are a [Visual Data Specialist]. Your task is to analyze complex questions and create a concise logical strategy to find the answer from images and tables.

You need to strictly follow the steps and complete them step by step through chatting.

**Step 1: Goal Alignment**
Analyze the user's question and define a concise execution strategy.
- Identify the exact anchors required (e.g., specific dates, numerical columns, row headers, or trend patterns).
- Outline the necessary operations (e.g., cross-referencing tables, calculating sums, or identifying extrema).
- **Constraints**: Use professional, data-driven language. Avoid numbered lists, restating the question, or conversational filler. Output must be a single, compact paragraph.
Wrap your plan within <plan></plan> tags.

##########
-Examples-
Question: How many % of Rep/Lean Rep people think cases have risen primally because of more testing and how many % of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy?
<plan>Locate 'Rep/Lean Rep' metrics across two thematic tables: extract the percentage for 'more testing' from the causality data and the percentage for 'federal government' from the policy responsibility data to synthesize the dual-metric response.</plan>
##########
Question: Of the four missions of Indian Space Program's space mission from 2012 to 2017, which mission includes the launch of least number of satellites?
<plan>Identify the four mission categories in the 2012–2017 table, count the specific satellite entries within each category, and perform a comparative analysis to isolate the category with the minimum aggregate count.</plan>
##########
Question: In how many hours Airbus incorporated a pop-up notification acknowledging the incident?
<plan>Scan the temporal sequence of UI mockups to pinpoint the first occurrence of the Airbus.com incident acknowledgment pop-up and extract the corresponding 'Hour' label as the definitive timestamp.</plan>
##########
Question: When did the number of tweets referencing Germanwings exceed 700,000?
<plan>Trace the 'Germanwings' tweet volume across the time-series charts to identify the specific chronological point where the quantitative value first surpasses the 700,000 threshold.</plan>
##########

Question: {query}
"""

PROMPTS["generate_step_2"] = """You are now switching to the role of [Visual Information Extractor]. Your task is to scan the provided images sequentially and extract objective facts aligned with the defined <plan>.

**Step 2: Information Collection**
Process images in numerical order, starting strictly with <record image="1">. For each, generate a record in this format:
- Only extract relevant data points, text, or visual information addressing the <plan> and question.
- If no relevant data is found, state "No relevant information."
- End the record with a judgment based on the following logic:
    - `Judge: Insufficient information` (if the question remains partially or fully unanswered).
    - `Judge: Sufficient information` (if the question can now be fully answered).
- If the judgment is `Judge: Insufficient information`, you must **CONTINUE COLLECT** information in the next image.
- If the judgment is `Judge: Sufficient information`, you must **IMMEDIATELY STOP** and do not process any further images.
Wrap information within <record image="N"></record> tags (where N is the image index).

##########
-Examples-
##########
<record image="1">Table shows for Rep/Lean Rep, 62% say "More people are being tested than in previous months" as the primary reason for rise in confirmed cases. Judge: Insufficient information</record>
<record image="2">No relevant information. Judge: Insufficient information</record>
<record image="3">Table confirms for Rep/Lean Rep, 62% say "More people are being tested than in previous months" as the primary reason. Judge: Insufficient information</record>
<record image="4">Table shows for Rep/Lean Rep, 30% say "The federal government" should be primarily responsible for developing and executing policies to limit the spread of COVID-19. Judge: Sufficient information</record>
##########
<record image="1">No relevant information. Judge: Insufficient information</record>
<record image="2">The pop-up window labeled “Airbus.com Hour 3” shows that Airbus displayed an incident acknowledgment pop-up at Hour 3. Judge: Sufficient information</record>
##########
<record image="1">No relevant information. Judge: Insufficient information</record>
<record image="2">No relevant information. Judge: Insufficient information</record>
<record image="3">No relevant information. Judge: Insufficient information</record>
<record image="4">No relevant information. Judge: Insufficient information</record>
<record image="5">No relevant information. Judge: Insufficient information</record>
##########

Question and images are provided below.
Question: {query}
"""

PROMPTS["generate_step_3"] = """You are now the [Logical Reasoning and Synthesis Officer]. Please produce the final answer based on the relevant information collected in Step 2.

**Step 3: Logical Synthesis & Reasoning**
Synthesize the collected information from Step 2.
- Perform step-by-step reasoning, calculations, or comparisons as defined in your <plan>.
- Cite the specific images used (e.g., "From image 1...") in your reasoning.
- If the loop finished without sufficient information, state that the information is missing.
- **STRICT PROHIBITION**: Do not mention "reference answer", "provided answer", "internal calibration", or suggest that you are following a hint. Your reasoning must appear 100% derived from the image records.
Wrap reasoning within <reason></reason> tags.

**Step 4: Answer Question**
Provide the final answer directly based on Step 3.
- Do NOT provide extra context or conversational filler.
- If information is missing, output <answer>Not answerable</answer>.
Wrap the final answer within <answer></answer> tags.

##########
-Examples-
##########
<reason>From image 1 and image 3, 62% of Rep/Lean Rep people think cases have risen primarily because of more testing. From image 4, 30% of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy. Both information are answered with direct data from the tables.</reason>
<answer>62% of Rep/Lean Rep people think cases have risen primarily because of more testing, and 30% of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy.</answer>
##########
<reason>From image 1: The satellite counts are Earth Observation: 5, Communication & Navigation: 19, Space Science & Planetary Exploration: 2, and Technology Development: 0 (not a satellite-launching category). Comparing the totals, among mission categories that involve satellites, Space Science & Planetary Exploration has the fewest satellite launches with 2.</reason>
<answer>Space Science & Planetary Exploration Satellites</answer>
##########
<reason>From image 2: The pop-up acknowledging the incident appears at Hour 3, clearly shown in the bottom-right image labeled “Airbus.com Hour 3”.</reason>
<answer>3 hours</answer>
##########
<reason>No information was collected from the provided inputs to determine when the Germanwings tweet count exceeded 700,000.</reason>
<answer>Not answerable</answer>
##########
Question: {query}
(Internal Calibration: {answer}. **WARNING**: This is for internal validation only. You MUST NOT mention this information or its existence in your response. Your reasoning must rely EXCLUSIVELY on the image records.)
"""
