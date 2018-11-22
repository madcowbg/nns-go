# nns-go
Experimental code to fit Neural Networks in Go... easily callable from other languages via JSON and the piping stdin/stdout.

## Why?

Wanted to play around with algorigthms for fitting neural networks. Python is ... so slow ... but great for visualisation, rapid prototyping, etc ... so this project was born.

The code is in Go, and requests/respones are JSON, so other languages can call it and use the fitting algorithms.

## How to use

Send JSON-formatted queries via STDIN:
```json
{"Order": {"D":2,"M":4,"K":3}, "Wts":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "ShouldFit": true}
```
and you will receive back the results in STDOUT:
```JSON
{
  "Wts": [1.161681348409446,1.161681348409446,1.161681348409446,1.161681348409446,1.161681348409446,1.161681348409446,1.161681348409446,1.161681348409446,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859,1.095318609619859],
  "Predicted":[[1,1,1]],
  "ErfValue":0,
  "Gradient":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "Hidden": [[0.22824409062744302,0.22824409062744302,0.22824409062744302,0.22824409062744302]]}
```
with a tintsy-wintsy bit of logging trash on STDERR.

*Note: Remember to flush the buffers...*

Newline characters in input and output should be fine (as part of well-formatted JSON), but be careful.

## Parameters
Right ... should include it soon. In the meantime, use the source to figure them out.