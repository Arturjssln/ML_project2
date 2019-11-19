
#### Here are some observations/interrogations from the PCA:
- When normalizing => mean got to 0, then some pixel are negative (display image has no more after normalizing image). But not information is lost (right? or information is lost while dividing by std_deviation?)
- We can see that we reduce the components from a lot (~400k) to less than 20 almost without lost => lot faster to do cNN
- Is it useful to apply PCA on our data or not finally ?
