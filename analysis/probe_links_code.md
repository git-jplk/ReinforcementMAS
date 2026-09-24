```
probe_links.py  domain=code  HUB=/hkfs/work/workspace/scratch/ma_jkliem-masterarbeit/hf_cache/hub

==============================================================================
[1] Forward der Links (modeling.py) - wo sitzen ln_source / ln_target?
==============================================================================
--- Adapter.forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        return self.post_ln(out)

--- CrossModelAdapter.forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = out + self.residual_proj(x)
        return self.ln_target(out)

LESEHILFE: Steht ln_target als LETZTE Operation (auf der Summe aus Residual
und MLP-Zweig), legt ihr Gain die Ausgabeskala fest (Variante A). Wird erst
ln_target(zweig) gebildet und DANACH das Residual addiert, haengt die Skala
weiterhin vom Input ab (Variante B).

==============================================================================
[2] Positionskodierung der Basismodelle
==============================================================================
--- planner  google/gemma-3-4b-it   model_type=gemma3_text  hidden=2560
      max_position_embeddings = 131072
      rope_parameters = {'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0}, 'full_attention': {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1000000.0}}
      layer_types = {'sliding_attention': 29, 'full_attention': 5}
--- critic   meta-llama/Llama-3.2-3B-Instruct   model_type=llama  hidden=3072
      max_position_embeddings = 131072
      rope_parameters = {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3', 'rope_theta': 500000.0}
--- solver   Qwen/Qwen3.5-4B   model_type=qwen3_5_text  hidden=2560
      max_position_embeddings = 262144
      rope_parameters = {'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'default', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25}
      layer_types = {'linear_attention': 24, 'full_attention': 8}
LESEHILFE: rope_theta vorhanden -> RoPE. Enthaelt layer_types neben
'full_attention' auch 'linear_attention' o. ae., ist das Modell ein Hybrid und
RoPE wirkt nur in einem Teil der Schichten.

==============================================================================
Embedding-Normen der Basismodelle (Referenz fuer [3])
==============================================================================
Testtext: MATH-500 (200 Aufgaben)
--- planner  key=language_model.model.embed_tokens.weight  Skalierung im Forward: x50.60
      alle Vokabel-Zeilen (roh)   : median    0.999 | p5    0.973 | p95    1.023 | mean    0.998
      Text-Tokens (wie eingespeist): median   51.480 | p5   49.824 | p95   52.747 | mean   51.401   n=1412
--- critic   key=model.embed_tokens.weight  Skalierung im Forward: x1.00
      alle Vokabel-Zeilen (roh)   : median    1.091 | p5    0.902 | p95    1.267 | mean    1.086
      Text-Tokens (wie eingespeist): median    1.085 | p5    0.852 | p95    1.246 | mean    1.068   n=1608
--- solver   key=model.language_model.embed_tokens.weight  Skalierung im Forward: x1.00
      alle Vokabel-Zeilen (roh)   : median    0.652 | p5    0.552 | p95    0.776 | mean    0.656
      Text-Tokens (wie eingespeist): median    0.642 | p5    0.545 | p95    0.786 | mean    0.651   n=1456

==============================================================================
InnerLinks (code) - Ausgabeskala (post_ln am Ende)
==============================================================================
--- planner  adapter(code).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 5.481
      Ausgabenorm InnerLink: median   50.239 | p5   49.698 | p95   50.869 | mean   50.256
      -> Rueckkopplung ins eigene Modell:   0.98x so gross wie normale Token-Embeddings
--- critic   adapter(code).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 8.185
      Ausgabenorm InnerLink: median   55.300 | p5   53.318 | p95   56.329 | mean   55.129
      -> Rueckkopplung ins eigene Modell:  50.99x so gross wie normale Token-Embeddings
--- solver   adapter(code).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 8.609
      Ausgabenorm InnerLink: median   49.498 | p5   46.519 | p95   51.458 | mean   49.350
      -> Rueckkopplung ins eigene Modell:  77.07x so gross wie normale Token-Embeddings

==============================================================================
OuterLinks (code) - Skala [3], Geometrie [4], Residual vs. Zweig [5], W3 [6]
==============================================================================
Stellvertreter fuer die echten Eingaben (= InnerLink-Ausgaben des Senders):
  skalentreu : post_ln_Sender(Token-Embedding)  -> gleiche Groesse wie echte Latents
  roh        : Token-Embedding wie eingespeist  -> nur zum Vergleich (v1)

--- Planner-Critic-Outerlink(code).pt   planner -> critic
  [skalentreu]  Eingang |x| median   50.239 | p5   49.698 | p95   50.869 | mean   50.256
      [3] Ausgang |y| median   55.425 | p5   55.425 | p95   55.425 | mean   55.425   Empfaenger-Token median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
          -> Ausgang / Empfaenger-Embedding:  51.10x
      [4] Geometrie r  0.931   1-NN erhalten  83.8 %   10-NN  79.2 %   mittl. Kosinus 0.025 -> 0.037
      [5] |Residual W3 x|    31.772   |MLP-Zweig|    10.961   Zweig/Residual     0.34
  [roh]  Eingang |x| median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
      [3] Ausgang |y| median   55.425 | p5   55.425 | p95   55.425 | mean   55.425   Empfaenger-Token median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
          -> Ausgang / Empfaenger-Embedding:  51.10x
      [4] Geometrie r  0.933   1-NN erhalten  83.0 %   10-NN  79.1 %   mittl. Kosinus 0.038 -> 0.049
      [5] |Residual W3 x|    32.534   |MLP-Zweig|    10.949   Zweig/Residual     0.34
  [6] W3 Singulaerwerte max 1.209 median 0.534 min 0.0554  Kondition 21.8  Band [0.8,1.25]xMedian 24.6 %

--- Critic-Solver-Outerlink(code).pt   critic -> solver
  [skalentreu]  Eingang |x| median   55.300 | p5   53.318 | p95   56.329 | mean   55.129
      [3] Ausgang |y| median   51.693 | p5   51.530 | p95   51.827 | mean   51.688   Empfaenger-Token median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
          -> Ausgang / Empfaenger-Embedding:  80.49x
      [4] Geometrie r  0.728   1-NN erhalten   7.8 %   10-NN  21.4 %   mittl. Kosinus 0.041 -> 0.763
      [5] |Residual W3 x|   420.173   |MLP-Zweig|  9648.786   Zweig/Residual    22.96
  [roh]  Eingang |x| median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
      [3] Ausgang |y| median   51.522 | p5   51.381 | p95   51.671 | mean   51.525   Empfaenger-Token median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
          -> Ausgang / Empfaenger-Embedding:  80.23x
      [4] Geometrie r  0.617   1-NN erhalten   9.0 %   10-NN  21.2 %   mittl. Kosinus 0.027 -> 0.692
      [5] |Residual W3 x|     6.068   |MLP-Zweig|  5516.550   Zweig/Residual   909.09
  [6] W3 Singulaerwerte max 39.725 median 0.715 min 0.0736  Kondition 540.0  Band [0.8,1.25]xMedian 22.8 %

--- Solver-Planner-Outerlink(code).pt   solver -> planner
  [skalentreu]  Eingang |x| median   49.498 | p5   46.519 | p95   51.458 | mean   49.350
      [3] Ausgang |y| median   50.596 | p5   50.595 | p95   50.596 | mean   50.596   Empfaenger-Token median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
          -> Ausgang / Empfaenger-Embedding:   0.98x
      [4] Geometrie r  0.964   1-NN erhalten  78.1 %   10-NN  73.5 %   mittl. Kosinus 0.029 -> 0.043
      [5] |Residual W3 x|    28.568   |MLP-Zweig|    10.009   Zweig/Residual     0.35
  [roh]  Eingang |x| median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
      [3] Ausgang |y| median   50.590 | p5   50.589 | p95   50.590 | mean   50.590   Empfaenger-Token median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
          -> Ausgang / Empfaenger-Embedding:   0.98x
      [4] Geometrie r  0.890   1-NN erhalten  69.5 %   10-NN  63.5 %   mittl. Kosinus 0.020 -> 0.130
      [5] |Residual W3 x|     0.683   |MLP-Zweig|     9.622   Zweig/Residual    14.09
  [6] W3 Singulaerwerte max 1.151 median 0.467 min 0.0001  Kondition 8701.8  Band [0.8,1.25]xMedian 21.1 %

==============================================================================
Kurzfassung (Geometrie/Residual: skalentreuer Stellvertreter, roh in Klammern)
==============================================================================
OuterLink           Skala        Geometrie r        1-NN erhalten       Zweig/Residual
Planner-Critic     51.10x    0.931 ( 0.933)      83.8 % ( 83.0 %)      0.34 (   0.34)
Critic-Solver      80.49x    0.728 ( 0.617)       7.8 % (  9.0 %)     22.96 ( 909.09)
Solver-Planner      0.98x    0.964 ( 0.890)      78.1 % ( 69.5 %)      0.35 (  14.09)

LESEHILFE
  Skala ~1x        -> Ausgaben passen zur Embedding-Skala des Empfaengers
  Skala >>1 / <<1  -> Latents sind lauter/leiser als normale Tokens (Folie 11)
  Geometrie r ~1   -> Nachrichtenstruktur bleibt erhalten; deutlich kleiner
                      stuetzt die Verzerrungs-Hypothese (Folie 12)
  Einschraenkung: Eingaben sind Token-Embeddings als Stellvertreter fuer
  InnerLink-Ausgaben - die Bestaetigung mit echten Latents braucht die GPU.
```
