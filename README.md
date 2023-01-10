# CS-E4740 - Federated Learning 
## course offered during spring 2023 at [Aalto University](https://www.aalto.fi/en) and to adult learners via [Finnish Network University](https://fitech.io/en/)

<a href="Lectures.md"> **Lectures** </a>   *** <a href="FL_LectureNotes.pdf"> **Lecture Notes** </a> *** <a href="Assignments.md"> **Assignments** </a>

## Abstract

Many application domains of machine learning (ML), such as numerical weather prediction, generate decentralized 
collections of local datasets. A naive application of basic ML methods [[1]](#1) would require collecting these local datasets 
at a central point. However, this approach might be unfavourable for several reasons, including inefficient use of 
computational infrastructure or the lack of privacy.

Federated learning (FL) aims at training ML models in a decentralized and collaborative fashion. 
FL methods require only the exchange of model parameter updates instead of raw data. These methods 
are appealing computationally and from a privacy protection perspective. Indeed, FL methods leverage 
distributed computational resources and minimize the leakage of private information irrelevant 
to the learning task.

This course uses an optimization perspective to study some widely used FL models and algorithms. In particular, we will 
discuss total variation minimization as a unifying design principle for FL methods. The geometry of total variation 
determines these methods' computational and statistical properties. 

## References
<a id="1">[1]</a> 
A. Jung, "Machine Learning. The Basics," Springer, Singapore, 2022. available via Aalto library [here](https://primo.aalto.fi/discovery/search?query=any,contains,machine%20learning%20the%20basics&tab=LibraryCatalog&search_scope=MyInstitution&vid=358AALTO_INST:VU1&lang=en&offset=0). [preprint](https://mlbook.cs.aalto.fi). 

## Additional Material

- [Federated learning: Basics and application to the mobile keyboard (ML Tech Talks)](https://www.youtube.com/watch?v=IXI1AjimfmE)
- [Network Flows in Federated Learning (slides for an IEEE Seminar Talk)](/slides/IEEE_Finland_CSS_RAS_SMCS.pdf)