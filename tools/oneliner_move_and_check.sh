#!/usr/bin/env bash

## move the Seq_* dir upwards
for d in Seq_*; do [ -d "$d/$d" ] && mv "$d/$d"/* "$d/" && rmdir "$d/$d"; done

### Check how many wc
for d in Seq_*; do echo -n "$d: "; ls -1 "$d" | wc -l; done

