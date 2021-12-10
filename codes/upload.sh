#! /bin/bash

# lists all the git commands required to push the changes to online repository.

git pull

# remove any python compiled file.

rm *.pyc

# remove all the swap files. ( this might also remove recovery files. so 
# proceed with caution ).

rm .*swp
rm .*swp
rm .*swo

# add all the files which has been changed.

git add --a

# commit all the added files.

git commit -a

# push all the files to the github

git push
