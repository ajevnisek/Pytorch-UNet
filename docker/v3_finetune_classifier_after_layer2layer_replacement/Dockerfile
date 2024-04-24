FROM ajevnisek/snl-amir:v0
COPY . ./
RUN pip install ipdb wandb
CMD ["/bin/bash", "-c", "source $EXPORT_SCRIPT && source $RUN_SCRIPT"]