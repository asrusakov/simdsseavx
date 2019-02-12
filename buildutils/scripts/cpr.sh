#!/bin/bash

for i in *.cpp *h ; do
  if ! grep -q Copyright $i
  then
    cat copyright.txt $i >$i.new && mv $i.new $i
  fi
done

for i in *.py ; do
  if ! grep -q Copyright $i
  then
    cat copyrightPy.txt $i >$i.new && mv $i.new $i
  fi
done

