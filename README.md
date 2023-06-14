# Exploring and Building  Visual Question Answering Systems using CLEVR and EasyVQA

Research Report: [Exploring and Building Visual Question Answering Systems using CLEVR and EasyVQA](https://github.com/iamsashank09/multimodal-visual-question-answering/blob/main/Developing%20VQA%20Systems.pdf)

Example 1:

![enter image description here](https://github.com/iamsashank09/multimodal-visual-question-answering/blob/main/Sample%20Outputs/Sample-2.jpg)

Example 2:

![enter image description here](https://github.com/iamsashank09/multimodal-visual-question-answering/blob/main/Sample%20Outputs/Sample-3.jpg)

**Please follow the below instructions to run our code:**

1) Download our best performing model checkpoint from [here](https://github.com/iamsashank09/multimodal-visual-question-answering/blob/main/models/best_val.model) and place in: 

		models/
2) Download the CLEVR dataset from [here](https://cs.stanford.edu/people/jcjohns/clevr/), we've used CLEVR v1.0 Main (Not CoGenT), place the data in:
		
		CLEVR_v1.0/
			images/
			questions/
	
	If you want to instead try it on a simpler, smaller datatset you can try [EasyVQA](https://github.com/vzhou842/easy-VQA), which has only 13 classes. The code and process remains the same. 

3) Run *[multimodel-clevr-public.py](https://github.com/iamsashank09/multimodal-visual-question-answering/blob/main/multimodel-clevr-public.py "multimodel-clevr-public.py")* which is in the project root folder:
(Make sure all the requirements are installed)

	    python multimodel-clevr-public.py
