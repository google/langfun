# Langfun

Langfun is a PyGlove powered library that aims to make language models (LM) fun
to work with. Its design enables a programming model that flows naturally,
resembling the human thought process. It emphasizes the reuse and combination of
language pieces to form prompts, thereby accelerating innovation. In contrast to
other LM frameworks, which feed program-generated data into the LM, langfun
takes a distinct approach: It starts with natural language, allowing for
seamless interactions between language and program logic, and concludes with
natural language and optional structured output. Consequently, langfun can
aptly be described as Language as functions, capturing the core of its
methodology.


# Hello World

```python
import langfun as lf

class NumericAnswerExtractor(lf.LangFunc):
  """Numeric answer extractor.

  Here is my question:
  {{question}}

  Here is the response:
  {{question()}}

  Can you help me extract a number from the response as the answer to my
  question? Your response should only contain a number in numeric form.
  If the answer is not a number or you cannot extract it, respond with UNKNOWN.
  """
  output_transform = lf.transforms.Match('\d+').to_int()

l = NumericAnswerExtractor()

with lf.context(lm=lf.llms.Gpt35(debug=True)):
  r = l(question=lf.LangFunc('What is result of {{x}} plus {{y}}?'),
        x='one',
        y='two')
  print('Result:', r.result)
```
