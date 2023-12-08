# CS-E4740 - Federated Learning 
## course offered during spring 2024 at [Aalto University](https://www.aalto.fi/en) and to adult learners via [Finnish Network University](https://fitech.io/en/)

You can formally enrol this course as 
- university (of applied science) student in Finland (contact your study administrator for details) 
- adult learner in Finland via <a href="https://fitech.io/en/studies/federated-learning/">  FiTech  </a> 
- student at Chalmers, KTH (Sweden), NTNU (Norway), DTU (Denmark) via <a href="https://forms.gle/uSK2Uw71aLVqnymWA"> **Registration for Externals** </a> 


<a href="material/Lectures.md"> **Lectures** </a>   *** <a href="material/FL_LectureNotes.pdf"> **Lecture Notes** </a> *** <a href="material/Assignments.md"> **Assignments** </a> *** <a href="material/Studentproject.md"> **FL Project** </a>

## Abstract


Many machine learning (ML) application domains, such as numerical weather prediction, generate decentralized 
collections of local datasets. A naive application of basic ML methods [[1]](#1) would require collecting these local datasets 
at a central point. However, this approach might be unfavourable for several reasons, including inefficient use of 
computational infrastructure or the need for more privacy.

Federated learning (FL) aims to train ML models in a decentralized and collaborative fashion. 
FL methods require only the exchange of model parameter updates instead of raw data. These methods 
are appealing computationally and from a privacy protection perspective. Indeed, FL methods leverage 
distributed computational resources and minimize the leakage of private information irrelevant 
to the learning task.

This course teaches you how to apply concepts from linear algebra (arrays of numbers)
 and calculus (smooth curves) to analyze and design federated learning (FL) systems. You will
 learn to formulate "real-world" applications, ranging from high-precision weather forecasting
 to personalized health care, as optimization problems and solve them using distributed
 optimization algorithms. We offer the courses in a basic variant (5 credits) that you extend to an 
 extended variant (10 credits) by completing a student project. This student project allows you to 
 pilot (get feedback for) ideas for your thesis or current research.
 
 To get a more concrete idea of what to expect, have a look at the  <a href="material/FL_LectureNotes.pdf"> draft for the lecture notes. </a> 

## References
<a id="1">[1]</a> 
A. Jung, "Machine Learning. The Basics," Springer, Singapore, 2022. available via Aalto library [here](https://primo.aalto.fi/discovery/search?query=any,contains,machine%20learning%20the%20basics&tab=LibraryCatalog&search_scope=MyInstitution&vid=358AALTO_INST:VU1&lang=en&offset=0). [preprint](https://mlbook.cs.aalto.fi). 

## Additional Material

- [Federated learning: Basics and application to the mobile keyboard (ML Tech Talks)](https://www.youtube.com/watch?v=IXI1AjimfmE)
- [Network Flows in Federated Learning (slides for an IEEE Seminar Talk)](/slides/IEEE_Finland_CSS_RAS_SMCS.pdf)