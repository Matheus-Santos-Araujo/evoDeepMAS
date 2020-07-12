__includes [ "KmeansLuke.nls" ]
extensions [matrix array py r]

globals[
  nmto
  nto
  targets
  observers
  n-observers-in
  set-targets-min
  first-part
  second-part
  result-prediction
  neural-error
  err
  ntar-not-observed
  evasion
  generations
  inc
  genomes
  gen
  numero
  third-part
  step
  nsteps
  current-frame
  next-frame
  is-first-run
]

turtles-own[
  speed
  goal
  goalX
  goalY
  cluster-flag
  up-car?
]

to setup

   py:setup py:python3
  ca
  ask patches [ set pcolor white]
  ask patches []

   create-turtles number-observers
  [
    setxy random-xcor random-ycor
    set shape "person"
    set color blue
    set size 3
    set speed 1
    set goal 0
  ]

 create-turtles number-targets
 [
    setxy random-xcor random-ycor
    set shape "person"
    set color red
    set size 3
    set speed speed-targets
 ]

  set observers turtles with [color = blue]
  set targets turtles with [color = red]

  (py:run
    "from __future__ import print_function"
    "from evolver import Evolver"
    "from tqdm import tqdm"
    "import logging"
    "from keras import backend as K"
    "import sys"
    "import numpy as np"
    "from keras.models import load_model"
    "import h5py"
  )

  training

  reset-ticks
end

to training
  (py:run "model = load_model('modelo 6.h5')")
  (py:run "model.summary()")
end

to go

    if (ticks = time-experiment)
  [
    set nmto nto / ticks
    stop
  ]

  if (remainder ticks gamma = 0)
  [
    runkmeans
  ]

  runPredictionTargets

  ask turtles [
    if (color = blue)[
      behavior-observer
    ]
    if (color = red)[
      let nobs count observers with [distance myself <= range-sensor]

      if(nobs > 0)[
        set nto nto + 1
      ]

      bt
    ]
  ]

  tick
  set evasion ntar-not-observed / ticks
  set nmto nto / ticks
  set neural-error err / ticks
  go
end

to runkmeans

  let centers matrix:make-constant number-targets 2 -1
  let i 0

  let list-observers sort observers
  foreach list-observers[
    t ->

    matrix:set centers i 0 [xcor] of t
    matrix:set centers i 1 [ycor] of t

    set i i + 1
  ]

  let result runKmeansLuke centers 10 1
  ;execução kmeans


  ask observers[
    set goal 0
  ]
  ;atribuição dos destinos

  set i 0
  ;criando a lista de destinos


  foreach list-observers[
    o ->

    let x matrix:get result i 0
    let y matrix:get result i 1

    let p patch int x int y

    ask o[set goal p]



    set i i + 1
  ]

end

to runPredictionTargets

  let list-targets sort targets
  let x 0
  let y 0

  foreach list-targets
  [
    tar ->

    ask tar [

      let nearest-obs min-n-of 1 observers [distance myself]

      foreach sort nearest-obs [

        n-obs ->
          py:set "ObsPosX" [xcor] of n-obs
          py:set "ObsPosY" [ycor] of n-obs
          py:set "ObsOrien" [heading] of n-obs
          (py:run "inputPred = [ObsPosX, ObsPosY, ObsOrien]" )

          ; Quem são os 5 alvos mais próximos desse observador?
          ask n-obs [
            set set-targets-min min-n-of 5 targets [distance myself]

            foreach sort set-targets-min [
              agent ->

                py:set "TarPosX" ([xcor] of agent)
                py:set "TarPosY" ([ycor] of agent)
                (py:run
                "inputPred.append(TarPosX)"
                "inputPred.append(TarPosY)"
                )
            ]
        ]
        (py:run
          "inputPred = np.array(inputPred)"
      	  "inputPred = inputPred.reshape(1, 13)"
          "#print(inputPred.shape)"
          "result = model.predict(inputPred, batch_size = 1)"
          "result = tuple(result.reshape(1, -1)[0])"
          "if np.isnan(result).any():"
          "    result = (0, 0)"
        )
        set result-prediction py:runresult "result"
        ;print "fkjsadlçfjasp"
        ;print result-prediction

        ; Calculando o Erro (RMSE)

        let error-neural 0
        ask nearest-obs [
          set error-neural sqrt ((xcor - (first result-prediction)) * (xcor - (first result-prediction)) + (ycor - (last result-prediction)) * (ycor - (last result-prediction)))
          ;print("Resultado Real de X-------")
          ;print(xcor)
          ;print("Resultado da Predicao de X-------")
          ;print(first result-prediction)

          ;print("Resultado Real de Y----------")
          ;print(ycor)
          ;print("Resultado da Predicao de Y-------")
          ;print(last result-prediction)
        ]
        set err err + error-neural

        ; Fim do Calculo do Erro (RMSE)

        set x (x + (first result-prediction))
        set y (y + (last result-prediction))

      ]

    ]

    ask tar [ set goalX x ]
    ask tar [ set goalY y ]

    set x 0
    set y 0

  ]
end

to bt

  ifelse not can-move? 1
  [
    ;right random 360
    ;left random 360
    facexy 75 75
    fd speed-targets
  ] [
    facexy goalX goalY
    rt 180
    fd speed-targets
  ]


end

to behavior-observer
 face goal

  fd 1
end
@#$#@#$#@
GRAPHICS-WINDOW
439
10
897
469
-1
-1
3.0
1
10
1
1
1
0
0
0
1
0
149
0
149
0
0
1
ticks
30.0

SLIDER
41
34
213
67
number-observers
number-observers
1
24
12.0
1
1
NIL
HORIZONTAL

SLIDER
41
86
213
119
number-targets
number-targets
1
24
24.0
1
1
NIL
HORIZONTAL

BUTTON
248
151
319
184
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

INPUTBOX
243
13
398
73
time-experiment
400.0
1
0
Number

SLIDER
241
86
413
119
gamma
gamma
1
15
15.0
2
1
NIL
HORIZONTAL

BUTTON
332
151
395
184
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
248
198
315
239
NIL
NMTO
3
1
10

MONITOR
330
195
396
240
RMSE
(neural-error * 400) / 9600
3
1
11

PLOT
30
271
207
405
NMTO
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot nmto"

PLOT
229
272
406
407
RMSE
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot neural-error"

CHOOSER
56
137
194
182
speed-targets
speed-targets
0.1 0.25 0.5 0.75 0.9
0

CHOOSER
56
201
194
246
range-sensor
range-sensor
5 10 15 20 25
0

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

building store
false
0
Rectangle -7500403 true true 30 45 45 240
Rectangle -16777216 false false 30 45 45 165
Rectangle -7500403 true true 15 165 285 255
Rectangle -16777216 true false 120 195 180 255
Line -7500403 true 150 195 150 255
Rectangle -16777216 true false 30 180 105 240
Rectangle -16777216 true false 195 180 270 240
Line -16777216 false 0 165 300 165
Polygon -7500403 true true 0 165 45 135 60 90 240 90 255 135 300 165
Rectangle -7500403 true true 0 0 75 45
Rectangle -16777216 false false 0 0 75 45

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

car top
true
0
Polygon -7500403 true true 151 8 119 10 98 25 86 48 82 225 90 270 105 289 150 294 195 291 210 270 219 225 214 47 201 24 181 11
Polygon -16777216 true false 210 195 195 210 195 135 210 105
Polygon -16777216 true false 105 255 120 270 180 270 195 255 195 225 105 225
Polygon -16777216 true false 90 195 105 210 105 135 90 105
Polygon -1 true false 205 29 180 30 181 11
Line -7500403 false 210 165 195 165
Line -7500403 false 90 165 105 165
Polygon -16777216 true false 121 135 180 134 204 97 182 89 153 85 120 89 98 97
Line -16777216 false 210 90 195 30
Line -16777216 false 90 90 105 30
Polygon -1 true false 95 29 120 30 119 11

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

factory
false
0
Rectangle -7500403 true true 76 194 285 270
Rectangle -7500403 true true 36 95 59 231
Rectangle -16777216 true false 90 210 270 240
Line -7500403 true 90 195 90 255
Line -7500403 true 120 195 120 255
Line -7500403 true 150 195 150 240
Line -7500403 true 180 195 180 255
Line -7500403 true 210 210 210 240
Line -7500403 true 240 210 240 240
Line -7500403 true 90 225 270 225
Circle -1 true false 37 73 32
Circle -1 true false 55 38 54
Circle -1 true false 96 21 42
Circle -1 true false 105 40 32
Circle -1 true false 129 19 42
Rectangle -7500403 true true 14 228 78 270

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

house colonial
false
0
Rectangle -7500403 true true 270 75 285 255
Rectangle -7500403 true true 45 135 270 255
Rectangle -16777216 true false 124 195 187 256
Rectangle -16777216 true false 60 195 105 240
Rectangle -16777216 true false 60 150 105 180
Rectangle -16777216 true false 210 150 255 180
Line -16777216 false 270 135 270 255
Polygon -7500403 true true 30 135 285 135 240 90 75 90
Line -16777216 false 30 135 285 135
Line -16777216 false 255 105 285 135
Line -7500403 true 154 195 154 255
Rectangle -16777216 true false 210 195 255 240
Rectangle -16777216 true false 135 150 180 180

lander 2
true
0
Polygon -7500403 true true 135 205 120 235 180 235 165 205
Polygon -16777216 false false 135 205 120 235 180 235 165 205
Line -7500403 true 75 30 195 30
Polygon -7500403 true true 195 150 210 165 225 165 240 150 240 135 225 120 210 120 195 135
Polygon -16777216 false false 195 150 210 165 225 165 240 150 240 135 225 120 210 120 195 135
Polygon -7500403 true true 75 75 105 45 195 45 225 75 225 135 195 165 105 165 75 135
Polygon -16777216 false false 75 75 105 45 195 45 225 75 225 120 225 135 195 165 105 165 75 135
Polygon -16777216 true false 217 90 210 75 180 60 180 90
Polygon -16777216 true false 83 90 90 75 120 60 120 90
Polygon -16777216 false false 135 165 120 135 135 75 150 60 165 75 180 135 165 165
Circle -7500403 true true 120 15 30
Circle -16777216 false false 120 15 30
Line -7500403 true 150 0 150 45
Polygon -1184463 true false 90 165 105 210 195 210 210 165
Line -1184463 false 210 165 245 239
Line -1184463 false 237 221 194 207
Rectangle -1184463 true false 221 245 261 238
Line -1184463 false 90 165 55 239
Line -1184463 false 63 221 106 207
Rectangle -1184463 true false 39 245 79 238
Polygon -16777216 false false 90 165 105 210 195 210 210 165
Rectangle -16777216 false false 221 237 262 245
Rectangle -16777216 false false 38 237 79 245

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tile stones
false
0
Polygon -7500403 true true 0 240 45 195 75 180 90 165 90 135 45 120 0 135
Polygon -7500403 true true 300 240 285 210 270 180 270 150 300 135 300 225
Polygon -7500403 true true 225 300 240 270 270 255 285 255 300 285 300 300
Polygon -7500403 true true 0 285 30 300 0 300
Polygon -7500403 true true 225 0 210 15 210 30 255 60 285 45 300 30 300 0
Polygon -7500403 true true 0 30 30 0 0 0
Polygon -7500403 true true 15 30 75 0 180 0 195 30 225 60 210 90 135 60 45 60
Polygon -7500403 true true 0 105 30 105 75 120 105 105 90 75 45 75 0 60
Polygon -7500403 true true 300 60 240 75 255 105 285 120 300 105
Polygon -7500403 true true 120 75 120 105 105 135 105 165 165 150 240 150 255 135 240 105 210 105 180 90 150 75
Polygon -7500403 true true 75 300 135 285 195 300
Polygon -7500403 true true 30 285 75 285 120 270 150 270 150 210 90 195 60 210 15 255
Polygon -7500403 true true 180 285 240 255 255 225 255 195 240 165 195 165 150 165 135 195 165 210 165 255

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

tree pine
false
0
Rectangle -6459832 true false 120 225 180 300
Polygon -7500403 true true 150 240 240 270 150 135 60 270
Polygon -7500403 true true 150 75 75 210 150 195 225 210
Polygon -7500403 true true 150 7 90 157 150 142 210 157 150 7

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experimento1" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <metric>nmto</metric>
    <metric>neural-error</metric>
    <enumeratedValueSet variable="number-observers">
      <value value="12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="range-sensor">
      <value value="5"/>
      <value value="10"/>
      <value value="15"/>
      <value value="20"/>
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="speed-targets">
      <value value="0.1"/>
      <value value="0.25"/>
      <value value="0.5"/>
      <value value="0.75"/>
      <value value="0.9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-targets">
      <value value="24"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gamma">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="time-experiment">
      <value value="400"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="time-experiment">
      <value value="400"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
