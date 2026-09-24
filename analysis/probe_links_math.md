```
probe_links.py  domain=math  HUB=/hkfs/work/workspace/scratch/ma_jkliem-masterarbeit/hf_cache/hub

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
InnerLinks (math) - Ausgabeskala (post_ln am Ende)
==============================================================================
--- planner  adapter(math).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 5.529
      Ausgabenorm InnerLink: median   50.241 | p5   49.716 | p95   50.863 | mean   50.250
      -> Rueckkopplung ins eigene Modell:   0.98x so gross wie normale Token-Embeddings
--- critic   adapter(math).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 8.079
      Ausgabenorm InnerLink: median   55.214 | p5   53.373 | p95   56.200 | mean   55.063
      -> Rueckkopplung ins eigene Modell:  50.91x so gross wie normale Token-Embeddings
--- solver   adapter(math).pt   post_ln gain: median    1.000 | p5    1.000 | p95    1.000 | mean    1.000   |bias| 8.743
      Ausgabenorm InnerLink: median   49.630 | p5   46.553 | p95   51.556 | mean   49.429
      -> Rueckkopplung ins eigene Modell:  77.28x so gross wie normale Token-Embeddings

==============================================================================
OuterLinks (math) - Skala [3], Geometrie [4], Residual vs. Zweig [5], W3 [6]
==============================================================================
Stellvertreter fuer die echten Eingaben (= InnerLink-Ausgaben des Senders):
  skalentreu : post_ln_Sender(Token-Embedding)  -> gleiche Groesse wie echte Latents
  roh        : Token-Embedding wie eingespeist  -> nur zum Vergleich (v1)

--- Planner-Critic-Outerlink(math).pt   planner -> critic
  [skalentreu]  Eingang |x| median   50.241 | p5   49.716 | p95   50.863 | mean   50.250
      [3] Ausgang |y| median   55.425 | p5   55.425 | p95   55.425 | mean   55.425   Empfaenger-Token median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
          -> Ausgang / Empfaenger-Embedding:  51.10x
      [4] Geometrie r  0.931   1-NN erhalten  83.6 %   10-NN  79.3 %   mittl. Kosinus 0.025 -> 0.037
      [5] |Residual W3 x|    31.763   |MLP-Zweig|    10.951   Zweig/Residual     0.34
  [roh]  Eingang |x| median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
      [3] Ausgang |y| median   55.425 | p5   55.425 | p95   55.425 | mean   55.425   Empfaenger-Token median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
          -> Ausgang / Empfaenger-Embedding:  51.10x
      [4] Geometrie r  0.933   1-NN erhalten  82.8 %   10-NN  79.2 %   mittl. Kosinus 0.038 -> 0.049
      [5] |Residual W3 x|    32.532   |MLP-Zweig|    10.949   Zweig/Residual     0.34
  [6] W3 Singulaerwerte max 1.209 median 0.534 min 0.0556  Kondition 21.7  Band [0.8,1.25]xMedian 24.6 %

--- Critic-Solver-Outerlink(math).pt   critic -> solver
  [skalentreu]  Eingang |x| median   55.214 | p5   53.373 | p95   56.200 | mean   55.063
      [3] Ausgang |y| median   51.116 | p5   50.929 | p95   51.301 | mean   51.112   Empfaenger-Token median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
          -> Ausgang / Empfaenger-Embedding:  79.60x
      [4] Geometrie r  0.749   1-NN erhalten  19.5 %   10-NN  29.8 %   mittl. Kosinus 0.039 -> 0.689
      [5] |Residual W3 x|   192.599   |MLP-Zweig|  1267.894   Zweig/Residual     6.58
  [roh]  Eingang |x| median    1.085 | p5    0.852 | p95    1.246 | mean    1.068
      [3] Ausgang |y| median   50.939 | p5   50.788 | p95   51.193 | mean   50.953   Empfaenger-Token median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
          -> Ausgang / Empfaenger-Embedding:  79.32x
      [4] Geometrie r  0.690   1-NN erhalten  20.7 %   10-NN  34.3 %   mittl. Kosinus 0.027 -> 0.492
      [5] |Residual W3 x|     3.406   |MLP-Zweig|   419.877   Zweig/Residual   123.29
  [6] W3 Singulaerwerte max 13.624 median 0.971 min 0.0944  Kondition 144.3  Band [0.8,1.25]xMedian 20.7 %

--- Solver-Planner-Outerlink(math).pt   solver -> planner
  [skalentreu]  Eingang |x| median   49.630 | p5   46.553 | p95   51.556 | mean   49.429
      [3] Ausgang |y| median   50.596 | p5   50.595 | p95   50.596 | mean   50.596   Empfaenger-Token median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
          -> Ausgang / Empfaenger-Embedding:   0.98x
      [4] Geometrie r  0.965   1-NN erhalten  78.3 %   10-NN  73.2 %   mittl. Kosinus 0.032 -> 0.045
      [5] |Residual W3 x|    28.591   |MLP-Zweig|    10.000   Zweig/Residual     0.35
  [roh]  Eingang |x| median    0.642 | p5    0.545 | p95    0.786 | mean    0.651
      [3] Ausgang |y| median   50.589 | p5   50.589 | p95   50.590 | mean   50.589   Empfaenger-Token median   51.480 | p5   49.824 | p95   52.747 | mean   51.401
          -> Ausgang / Empfaenger-Embedding:   0.98x
      [4] Geometrie r  0.890   1-NN erhalten  69.7 %   10-NN  63.5 %   mittl. Kosinus 0.020 -> 0.130
      [5] |Residual W3 x|     0.683   |MLP-Zweig|     9.622   Zweig/Residual    14.09
  [6] W3 Singulaerwerte max 1.151 median 0.467 min 0.0001  Kondition 14405.0  Band [0.8,1.25]xMedian 21.1 %

==============================================================================
Kurzfassung (Geometrie/Residual: skalentreuer Stellvertreter, roh in Klammern)
==============================================================================
OuterLink           Skala        Geometrie r        1-NN erhalten       Zweig/Residual
Planner-Critic     51.10x    0.931 ( 0.933)      83.6 % ( 82.8 %)      0.34 (   0.34)
Critic-Solver      79.60x    0.749 ( 0.690)      19.5 % ( 20.7 %)      6.58 ( 123.29)
Solver-Planner      0.98x    0.965 ( 0.890)      78.3 % ( 69.7 %)      0.35 (  14.09)

LESEHILFE
  Skala ~1x        -> Ausgaben passen zur Embedding-Skala des Empfaengers
  Skala >>1 / <<1  -> Latents sind lauter/leiser als normale Tokens (Folie 11)
  Geometrie r ~1   -> Nachrichtenstruktur bleibt erhalten; deutlich kleiner
                      stuetzt die Verzerrungs-Hypothese (Folie 12)
  Einschraenkung: Eingaben sind Token-Embeddings als Stellvertreter fuer
  InnerLink-Ausgaben - die Bestaetigung mit echten Latents braucht die GPU.
```
