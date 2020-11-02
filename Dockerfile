FROM python:3.8
# based on debian

RUN apt update \
    && apt upgrade -y \
    && apt install -y \
        libcairo2-dev \
        ffmpeg \
        texlive \
        texlive-latex-extra \
        texlive-fonts-extra \
        texlive-latex-recommended \
        texlive-science \
        tipa \
        wget \
        gcc \
        nano \
        git


# clone manim
RUN git clone --depth 1 https://github.com/ManimCommunity/manim.git /opt/manim

# install dependencies
RUN pip3 install Pillow progressbar grpcio-tools grpcio pydub watchdog tqdm rich numpy pygments scipy colour pycairo cairocffi pangocffi pangocairocffi
# ensure ffi bindings are generated correctly
WORKDIR /usr/local/lib/python3.8/site-packages
RUN python cairocffi/ffi_build.py \
    && python pangocffi/ffi_build.py \
    && python pangocairocffi/ffi_build.py

WORKDIR /opt/manim

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]


# build with:
# docker build --label="manimce" --tag="manimce" .

# test option 1 with:
# docker run --rm -it manimce "apt install -y wget && wget 'https://raw.githubusercontent.com/ManimCommunity/manim/master/example_scenes/basic.py' && python -m manim basic.py SquareToCircle --low_quality -s && echo yay || echo nay"

# test the other options with:
# docker run --rm -it manimce "python -m manim example_scenes/basic.py SquareToCircle --low_quality -s > /dev/null 2>&1 && echo yay || echo nay"
