# AlchemyAO
This is my implementation of [The Alchemy Screen-space Ambient Obscurane Algorithm](https://research.nvidia.com/publication/alchemy-screen-space-ambient-obscurance-algorithm) from nVidia.
I was pretty happy with the results and definitely consider this as one of the best indirect illumination techniques that I've encountered so far.  I've also expanded my deferred shading framework 
to now include Physically-Based Rendering (PBR). 

## Features:
  *  Full configurable SSAO as described in original Alchemy paper and further enhanced by Morgan McGuire's Scalable Ambient Occlusion (SAO) algorithm
  *  Smooth AO produced by two bilateral filter passes, one horizontal and one vertical done in a compute shader.  Each pass uses 8 taps (8x8=64 samples) with Gaussian
  weights modulated by linear depth difference.
  *  Models are rendered using PBR materials (metalic, stone, plastic, etc)
  
![Alt Text](https://github.com/timurson/AlchemyAO/blob/master/Image1.PNG)
![Alt Text](https://github.com/timurson/AlchemyAO/blob/master/Image2.PNG)
![Alt Text](https://github.com/timurson/AlchemyAO/blob/master/Image3.PNG)
![Alt Text](https://github.com/timurson/AlchemyAO/blob/master/Image4.PNG)

## External Libraries Used:

[GLFW](https://www.glfw.org/download.html)
[GLAD](https://glad.dav1d.de/)
[GLM](https://glm.g-truc.net/0.9.8/index.html)
[Deam ImGui](https://github.com/ocornut/imgui)
[Assimp](http://assimp.org/index.php/downloads)
[STB](https://github.com/nothings)
[GLSW](https://prideout.net/blog/old/blog/index.html@p=11.html)

# License
Copyright (C) 2021 Roman Timurson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

