# TREEBERT2GEN
Tree transformer to decoder conditional event generation<br>
Made the following changes to the Tree transformer model:
- Changed the attention aggregation scheme in the proposed hierarchial attention scheme
- Supports word piece tokenization, original model only uses 1st sub-token of the word-piece split of a token. 
If the input is "Boy Jumped", let the word piece tokenization be "Bo #y Ju #mp #ed", then the original architecture could only support input as  "Bo Ju" .
- Change the intilaization scheme to allow incorporating prior knowledge about phrases (you don't want cross branching b/w two phrases)

