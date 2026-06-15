"""
Manim scenes para MLearning — compatibles con Manim 0.20, sin LaTeX.
Renderizar: python render_animations.py
"""
from manim import *
import numpy as np

config.tex_template = TexTemplate()
config.renderer = "cairo"

NAVY       = "#1e3a5f"
NAVY_LIGHT = "#4a7ec8"
GOLD       = "#f0a500"
SOFT_BG    = "#0d1b2a"


# ══════════════════════════════════════════════════════════════════════════════
#  LANDING — Linear → Clusters → Scatter
# ══════════════════════════════════════════════════════════════════════════════
class LandingAnimation(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 4, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.5},
            tips=False,
        ).scale(0.75)

        np.random.seed(42)
        xs = np.linspace(-2.5, 2.5, 30)
        ys = 0.8 * xs + 0.5 + np.random.randn(30) * 0.35

        dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.07, color=NAVY_LIGHT)
            for x, y in zip(xs, ys)
        ])
        line = axes.plot(lambda x: 0.8 * x + 0.5, color=GOLD, stroke_width=3)
        title1 = Text("Regresion Lineal", font_size=28, color=GOLD).to_edge(UP, buff=0.4)
        eq = Text("y = b0 + b1*x", font_size=22, color=WHITE).next_to(axes, DOWN)

        self.play(FadeIn(title1), Create(axes), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04), run_time=1.0)
        self.play(Create(line), Write(eq), run_time=0.9)
        self.wait(1.0)

        # → Clusters
        title2 = Text("K-Means Clustering", font_size=28, color=GOLD).to_edge(UP, buff=0.4)
        centers = [(-1.5, -0.5), (0.8, 1.2), (1.8, -1.0)]
        colors  = [BLUE_C, GREEN_C, RED_C]

        colored_dots = VGroup()
        for i, (cx, cy) in enumerate(centers):
            for _ in range(18):
                dx, dy = np.random.randn() * 0.38, np.random.randn() * 0.38
                colored_dots.add(Dot(axes.c2p(cx+dx, cy+dy), radius=0.07, color=colors[i]))

        centroids = VGroup(*[
            Star(n=5, outer_radius=0.18, color=GOLD, fill_opacity=1)
            .move_to(axes.c2p(cx, cy))
            for cx, cy in centers
        ])

        self.play(
            FadeOut(title1, line, eq),
            Transform(dots, colored_dots),
            FadeIn(title2),
            run_time=0.9,
        )
        self.play(LaggedStart(*[GrowFromCenter(c) for c in centroids], lag_ratio=0.2), run_time=0.7)
        self.wait(1.0)

        # → Scatter
        title3 = Text("Analisis Exploratorio", font_size=28, color=GOLD).to_edge(UP, buff=0.4)
        np.random.seed(7)
        scatter = VGroup(*[
            Dot(axes.c2p(np.random.uniform(-2.5, 2.5), np.random.uniform(-1.5, 3.0)),
                radius=0.07, color=np.random.choice([BLUE_C, GREEN_C, RED_C, PURPLE_C]))
            for _ in range(60)
        ])
        pca_arrow = Arrow(axes.c2p(-2, -1), axes.c2p(2, 2),
                          color=GOLD, stroke_width=4, max_tip_length_to_length_ratio=0.08)

        self.play(FadeOut(title2, centroids), Transform(dots, scatter), FadeIn(title3), run_time=0.9)
        self.play(GrowArrow(pca_arrow), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESIÓN LINEAL — Gradient Descent
# ══════════════════════════════════════════════════════════════════════════════
class LinearRegressionScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Regresion Lineal — Minimos Cuadrados", font_size=22, color=WHITE).to_edge(UP)
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 5, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.5},
            x_length=9, y_length=5, tips=False,
        )
        xlbl = Text("x", font_size=20, color=NAVY_LIGHT).next_to(axes.x_axis, RIGHT)
        ylbl = Text("y", font_size=20, color=NAVY_LIGHT).next_to(axes.y_axis, UP)

        np.random.seed(42)
        xs = np.linspace(-2.5, 2.5, 25)
        ys = 1.2 * xs + 0.8 + np.random.randn(25) * 0.45

        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.08, color=NAVY_LIGHT) for x, y in zip(xs, ys)])

        self.play(Write(title), Create(axes), Write(xlbl), Write(ylbl), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04), run_time=1.0)

        slope_t     = ValueTracker(-2.0)
        intercept_t = ValueTracker(3.0)

        line = always_redraw(lambda: axes.plot(
            lambda x: slope_t.get_value() * x + intercept_t.get_value(),
            color=GOLD, stroke_width=2.5, x_range=[-2.8, 2.8]
        ))

        # Ecuacion — fijada abajo a la izquierda
        eq = always_redraw(lambda: Text(
            f"y = {slope_t.get_value():.2f}x + {intercept_t.get_value():.2f}",
            font_size=20, color=GOLD
        ).move_to(LEFT * 4.8 + DOWN * 3.2))

        # MSE — fijado arriba a la derecha (lejos del eje)
        mse_label = always_redraw(lambda: Text(
            f"MSE = {np.mean((ys - (slope_t.get_value()*xs + intercept_t.get_value()))**2):.3f}",
            font_size=20, color=RED_B
        ).move_to(RIGHT * 5.2 + UP * 3.2))

        residuals = always_redraw(lambda: VGroup(*[
            Line(axes.c2p(x, y),
                 axes.c2p(x, slope_t.get_value()*x + intercept_t.get_value()),
                 color=RED_B, stroke_width=1.0, stroke_opacity=0.5)
            for x, y in zip(xs, ys)
        ]))

        self.play(Create(line), Write(eq), FadeIn(mse_label), Create(residuals))
        self.play(
            slope_t.animate.set_value(1.2),
            intercept_t.animate.set_value(0.8),
            run_time=3.5, rate_func=smooth,
        )
        self.wait(0.5)
        # Convergencia — abajo a la derecha, lejos de la ecuacion
        done = Text("Convergencia — MSE minimo", font_size=20, color=GREEN_C).move_to(RIGHT * 3.5 + DOWN * 3.2)
        self.play(FadeIn(done), line.animate.set_stroke(color=GREEN_C), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESIÓN LOGÍSTICA — Sigmoide
# ══════════════════════════════════════════════════════════════════════════════
class LogisticRegressionScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Regresion Logistica — Funcion Sigmoide", font_size=22, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-6, 6, 2], y_range=[-0.1, 1.2, 0.25],
            axis_config={"color": NAVY_LIGHT},
            x_length=9, y_length=4, tips=False,
        ).shift(DOWN * 0.3)

        sigmoid   = axes.plot(lambda x: 1 / (1 + np.exp(-x)), color=GOLD, stroke_width=3)
        threshold = DashedLine(
            axes.c2p(-6, 0.5), axes.c2p(6, 0.5),
            color=WHITE, stroke_width=1.5, dash_length=0.15
        )

        eq_text  = Text("sigma(z) = 1 / (1 + e^(-z))", font_size=22, color=GOLD).to_corner(UR)
        thr_text = Text("umbral = 0.5", font_size=16, color=WHITE).next_to(axes.c2p(3, 0.5), UP, buff=0.1)

        self.play(Create(axes), run_time=0.6)
        self.play(Create(sigmoid), Write(eq_text), run_time=1.0)
        self.play(Create(threshold), Write(thr_text), run_time=0.7)

        zone_neg = axes.get_area(sigmoid, x_range=[-6, 0], color=RED_C,   opacity=0.15)
        zone_pos = axes.get_area(sigmoid, x_range=[0,  6], color=GREEN_C, opacity=0.15)
        lbl_neg  = Text("Clase 0", font_size=18, color=RED_C).move_to(axes.c2p(-3.5, 0.2))
        lbl_pos  = Text("Clase 1", font_size=18, color=GREEN_C).move_to(axes.c2p(3.5, 0.8))

        self.play(FadeIn(zone_neg, zone_pos), Write(lbl_neg), Write(lbl_pos), run_time=0.8)

        x_t = ValueTracker(-5)
        dot = always_redraw(lambda: Dot(
            axes.c2p(x_t.get_value(), 1/(1+np.exp(-x_t.get_value()))),
            color=GOLD, radius=0.13
        ))
        prob = always_redraw(lambda: Text(
            f"P = {1/(1+np.exp(-x_t.get_value())):.3f}",
            font_size=22, color=GOLD
        ).to_corner(DR))

        self.play(FadeIn(dot, prob))
        self.play(x_t.animate.set_value(5), run_time=3.0, rate_func=smooth)
        self.wait(1.0)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  K-MEANS — Algoritmo EM
# ══════════════════════════════════════════════════════════════════════════════
class KMeansScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("K-Means — Algoritmo EM", font_size=22, color=WHITE).to_edge(UP)
        self.play(Write(title))

        # Axes desplazados a la derecha para dejar columna izquierda libre
        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.0},
            x_length=5.5, y_length=5.5, tips=False,
        ).shift(RIGHT * 2.0)
        self.play(Create(axes), run_time=0.5)

        np.random.seed(0)
        true_centers = [(-2, -2), (0, 2), (2.5, -1)]
        colors = [BLUE_C, GREEN_C, RED_C]
        all_pts = []
        for cx, cy in true_centers:
            for _ in range(20):
                all_pts.append((cx + np.random.randn()*0.5, cy + np.random.randn()*0.5))
        np.random.shuffle(all_pts)

        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.07, color=WHITE) for x, y in all_pts])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.02), run_time=1.0)

        init_centers = [(-1, 1), (1, -2), (3, 2)]
        centroids = VGroup(*[
            Star(n=6, outer_radius=0.22, color=colors[i], fill_opacity=1)
            .move_to(axes.c2p(cx, cy))
            for i, (cx, cy) in enumerate(init_centers)
        ])

        # --- Panel izquierdo: lista de pasos acumulativa ---
        panel_x = LEFT * 5.0
        panel_top = UP * 2.8
        log_header = Text("Registro de pasos", font_size=16, color=GOLD, weight=BOLD, font="Arial")\
            .move_to(panel_x + panel_top)
        separator = Line(
            panel_x + panel_top + DOWN * 0.35 + LEFT * 1.4,
            panel_x + panel_top + DOWN * 0.35 + RIGHT * 1.4,
            color=NAVY_LIGHT, stroke_width=1.0
        )
        self.play(FadeIn(log_header), Create(separator), FadeIn(centroids), run_time=0.7)

        init_entry = Text("Centroides iniciales", font_size=14, color=WHITE, font="Arial")\
            .move_to(panel_x + panel_top + DOWN * 0.85)
        self.play(FadeIn(init_entry), run_time=0.5)
        self.wait(0.8)

        def assign(centers):
            grp = VGroup()
            for x, y in all_pts:
                dists = [((x-cx)**2+(y-cy)**2)**0.5 for cx, cy in centers]
                grp.add(Dot(axes.c2p(x, y), radius=0.07, color=colors[np.argmin(dists)]))
            return grp

        def recompute(centers):
            new = []
            for i in range(3):
                pts = [(x, y) for x, y in all_pts
                       if np.argmin([((x-cx)**2+(y-cy)**2)**0.5 for cx, cy in centers]) == i]
                new.append((np.mean([p[0] for p in pts]) if pts else centers[i][0],
                            np.mean([p[1] for p in pts]) if pts else centers[i][1]))
            return new

        curr = init_centers
        log_entries = [init_entry]

        for it in range(3):
            offset_e = DOWN * (0.85 + len(log_entries) * 0.52)
            offset_m = DOWN * (0.85 + (len(log_entries) + 1) * 0.52)

            # E-step
            e_entry = Text(f"It.{it+1}  E-step: asignar clusters", font_size=13, color=BLUE_C, font="Arial")\
                .move_to(panel_x + panel_top + offset_e)
            self.play(FadeIn(e_entry), run_time=0.4)
            self.wait(0.8)
            self.play(Transform(dots, assign(curr)), run_time=1.0)
            self.wait(0.8)
            log_entries.append(e_entry)

            # M-step
            curr = recompute(curr)
            new_c = VGroup(*[
                Star(n=6, outer_radius=0.22, color=colors[i], fill_opacity=1)
                .move_to(axes.c2p(cx, cy))
                for i, (cx, cy) in enumerate(curr)
            ])
            m_entry = Text(f"It.{it+1}  M-step: mover centroides", font_size=13, color=GREEN_C, font="Arial")\
                .move_to(panel_x + panel_top + offset_m)
            self.play(FadeIn(m_entry), run_time=0.4)
            self.wait(0.8)
            self.play(Transform(centroids, new_c), run_time=0.9)
            self.wait(0.8)
            log_entries.append(m_entry)

        done_entry = Text("Convergencia alcanzada", font_size=14, color=GREEN_C, weight=BOLD, font="Arial")\
            .move_to(panel_x + panel_top + DOWN * (0.85 + len(log_entries) * 0.52))
        self.play(FadeIn(done_entry), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  PCA — Varianza y proyección
# ══════════════════════════════════════════════════════════════════════════════
class PCAScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("PCA — Maximizar Varianza Explicada", font_size=22, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            axis_config={"color": NAVY_LIGHT},
            x_length=8, y_length=5, tips=False,
        )
        np.random.seed(5)
        cov = np.array([[2.5, 1.8], [1.8, 1.5]])
        pts = np.random.multivariate_normal([0, 0], cov, 50)

        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.07, color=NAVY_LIGHT) for x, y in pts])
        self.play(Create(axes), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.03), run_time=1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        ev1 = eigenvectors[:, idx[0]] * 2.5
        ev2 = eigenvectors[:, idx[1]] * 1.0

        pc1 = Arrow(axes.c2p(-ev1[0], -ev1[1]), axes.c2p(ev1[0], ev1[1]),
                    color=GOLD, stroke_width=4, max_tip_length_to_length_ratio=0.08)
        pc2 = Arrow(axes.c2p(-ev2[0], -ev2[1]), axes.c2p(ev2[0], ev2[1]),
                    color=RED_C, stroke_width=3, max_tip_length_to_length_ratio=0.1)

        lpc1 = Text("PC1", font_size=24, color=GOLD).next_to(pc1.get_end(), UR, buff=0.1)
        lpc2 = Text("PC2", font_size=20, color=RED_C).next_to(pc2.get_end(), UL, buff=0.1)

        pct1 = eigenvalues[idx[0]] / eigenvalues.sum()
        pct2 = eigenvalues[idx[1]] / eigenvalues.sum()
        var_text = Text(
            f"PC1: {pct1:.0%} varianza\nPC2: {pct2:.0%} varianza",
            font_size=18, color=WHITE
        ).to_corner(DR)

        self.play(GrowArrow(pc1), Write(lpc1), run_time=0.8)
        self.play(GrowArrow(pc2), Write(lpc2), run_time=0.7)
        self.play(FadeIn(var_text), run_time=0.5)

        # Proyección sobre PC1
        proj_dots = VGroup()
        proj_lines = VGroup()
        for x, y in pts:
            t = (x * ev1[0] + y * ev1[1]) / (ev1[0]**2 + ev1[1]**2)
            px, py = t * ev1[0], t * ev1[1]
            proj_dots.add(Dot(axes.c2p(px, py), radius=0.07, color=GOLD, fill_opacity=0.8))
            proj_lines.add(DashedLine(axes.c2p(x, y), axes.c2p(px, py),
                                      stroke_width=0.8, color=GRAY, dash_length=0.08))

        lbl = Text("Proyeccion sobre PC1", font_size=18, color=GOLD).to_edge(DOWN)
        self.play(Write(lbl), Create(proj_lines), run_time=0.6)
        self.play(Transform(dots, proj_dots), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  SVM — Margen máximo
# ══════════════════════════════════════════════════════════════════════════════
class SVMScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("SVM — Margen Maximo", font_size=22, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            axis_config={"color": NAVY_LIGHT},
            x_length=8, y_length=5, tips=False,
        )
        self.play(Create(axes), run_time=0.5)

        np.random.seed(3)
        cls0 = np.random.multivariate_normal([-2, -1], [[0.4, 0], [0, 0.4]], 20)
        cls1 = np.random.multivariate_normal([2,  1],  [[0.4, 0], [0, 0.4]], 20)

        dots0 = VGroup(*[Dot(axes.c2p(x, y), radius=0.09, color=BLUE_C) for x, y in cls0])
        dots1 = VGroup(*[Dot(axes.c2p(x, y), radius=0.09, color=RED_C)  for x, y in cls1])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots0], lag_ratio=0.04),
            LaggedStart(*[GrowFromCenter(d) for d in dots1], lag_ratio=0.04),
            run_time=1.0,
        )

        boundary = axes.plot(lambda x: 0.5*x,       color=GOLD,      stroke_width=3,   x_range=[-3.5, 3.5])
        margin_p = DashedLine(axes.c2p(-3.5,  1.2+0.5*-3.5), axes.c2p(2.8,  1.2+0.5*2.8),
                              color=GOLD, stroke_width=1.5, dash_length=0.15)
        margin_n = DashedLine(axes.c2p(-2.8, -1.2+0.5*-2.8), axes.c2p(3.5, -1.2+0.5*3.5),
                              color=GOLD, stroke_width=1.5, dash_length=0.15)

        lboundary = Text("Hiperplano de decision", font_size=16, color=GOLD).next_to(axes.c2p(1, 1.8), RIGHT)
        lmargin   = Text("Margen", font_size=16, color=GOLD).next_to(axes.c2p(-2.5, 0.5), LEFT)

        self.play(Create(boundary), Write(lboundary), run_time=0.8)
        self.play(Create(margin_p), Create(margin_n), Write(lmargin), run_time=0.8)

        sv_circles = VGroup(
            Circle(radius=0.18, color=WHITE, stroke_width=2).move_to(axes.c2p(-1.1, -0.55)),
            Circle(radius=0.18, color=WHITE, stroke_width=2).move_to(axes.c2p(1.1,  0.55)),
        )
        sv_label = Text("Support Vectors", font_size=16, color=WHITE).to_edge(DOWN)
        self.play(Create(sv_circles), Write(sv_label), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
class DecisionTreeScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Arbol de Decision — Reglas if-else", font_size=22, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        # Nodos del árbol
        def node(txt, pos, color=NAVY_LIGHT, fs=18):
            box = RoundedRectangle(width=2.8, height=0.7, corner_radius=0.15,
                                   fill_color=color, fill_opacity=1, stroke_color=WHITE, stroke_width=1.5)
            box.move_to(pos)
            label = Text(txt, font_size=fs, color=WHITE, font="Arial").move_to(pos)
            return VGroup(box, label)

        root  = node("Edad > 30?",      UP * 1.5,             color="#1e3a5f")
        left  = node("Salario > 50k?",  LEFT * 3 + DOWN * 0.3, color="#2a5298")
        right = node("Clase: Alto",     RIGHT * 3 + DOWN * 0.3, color="#2e7d5a")
        ll    = node("Clase: Bajo",     LEFT * 4.5 + DOWN * 2.0, color="#c0392b", fs=16)
        lr    = node("Clase: Medio",    LEFT * 1.5 + DOWN * 2.0, color="#f0a500", fs=16)

        e1 = Line(root.get_bottom(), left.get_top(),  color=NAVY_LIGHT, stroke_width=2)
        e2 = Line(root.get_bottom(), right.get_top(), color=NAVY_LIGHT, stroke_width=2)
        e3 = Line(left.get_bottom(), ll.get_top(),    color=NAVY_LIGHT, stroke_width=1.5)
        e4 = Line(left.get_bottom(), lr.get_top(),    color=NAVY_LIGHT, stroke_width=1.5)

        lbl_si  = Text("Si",  font_size=14, color=GOLD, font="Arial").next_to(e1, UL, buff=0.05)
        lbl_no  = Text("No",  font_size=14, color=GOLD, font="Arial").next_to(e2, UR, buff=0.05)
        lbl_si2 = Text("Si",  font_size=14, color=GOLD, font="Arial").next_to(e3, UL, buff=0.05)
        lbl_no2 = Text("No",  font_size=14, color=GOLD, font="Arial").next_to(e4, UR, buff=0.05)

        # Panel derecho: métricas Gini
        panel_x = RIGHT * 4.8
        gini_title = Text("Impureza Gini", font_size=16, color=GOLD, font="Arial").move_to(panel_x + UP * 2.8)
        gini_eq    = Text("G = 1 - Σ pᵢ²", font_size=14, color=WHITE, font="Arial").move_to(panel_x + UP * 2.2)
        sep = Line(panel_x + UP * 1.9 + LEFT * 1.2, panel_x + UP * 1.9 + RIGHT * 1.2,
                   color=NAVY_LIGHT, stroke_width=1)
        gain_title = Text("Ganancia info.", font_size=15, color=GOLD, font="Arial").move_to(panel_x + UP * 1.5)
        gain_eq    = Text("G_padre - Σ G_hijos", font_size=13, color=WHITE, font="Arial").move_to(panel_x + UP * 1.0)

        self.play(FadeIn(gini_title), Write(gini_eq), Create(sep),
                  FadeIn(gain_title), Write(gain_eq), run_time=0.7)
        self.play(FadeIn(root), run_time=0.5)
        self.wait(0.5)
        self.play(Create(e1), Create(e2), Write(lbl_si), Write(lbl_no), run_time=0.6)
        self.play(FadeIn(left), FadeIn(right), run_time=0.5)
        self.wait(0.5)
        self.play(Create(e3), Create(e4), Write(lbl_si2), Write(lbl_no2), run_time=0.5)
        self.play(FadeIn(ll), FadeIn(lr), run_time=0.5)
        self.wait(1.0)

        # Highlight del camino de decisión
        path_dot = Dot(root.get_center(), color=GOLD, radius=0.15)
        self.play(FadeIn(path_dot))
        self.play(path_dot.animate.move_to(left.get_center()), run_time=0.8)
        self.play(path_dot.animate.move_to(lr.get_center()), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
class RandomForestScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Random Forest — Ensemble de Arboles", font_size=22, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        tree_colors = [BLUE_C, GREEN_C, GOLD]
        trees = VGroup()
        votes = VGroup()

        for i, col in enumerate(tree_colors):
            x = (i - 1) * 3.5
            trunk = Line(np.array([x, -2.5, 0]), np.array([x, -1.0, 0]), color=col, stroke_width=3)
            crown = VGroup(
                Circle(radius=0.7, fill_color=col, fill_opacity=0.7, stroke_width=0).move_to([x, -0.2, 0]),
                Circle(radius=0.5, fill_color=col, fill_opacity=0.5, stroke_width=0).move_to([x - 0.5, -0.6, 0]),
                Circle(radius=0.5, fill_color=col, fill_opacity=0.5, stroke_width=0).move_to([x + 0.5, -0.6, 0]),
            )
            lbl_tree = Text(f"Arbol {i+1}", font_size=14, color=col, font="Arial").move_to([x, -3.0, 0])
            trees.add(VGroup(trunk, crown, lbl_tree))

            # Bootstrap label
            bs = Text(f"Bootstrap {i+1}", font_size=12, color=WHITE, font="Arial").move_to([x, 1.2, 0])
            votes.add(bs)

        self.play(LaggedStart(*[GrowFromCenter(t) for t in trees], lag_ratio=0.3), run_time=1.2)
        self.wait(0.4)

        bs_title = Text("Muestras bootstrap (con reemplazo)", font_size=16, color=GOLD, font="Arial").move_to(UP * 2.0)
        self.play(Write(bs_title), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(v) for v in votes], lag_ratio=0.3), run_time=0.8)
        self.wait(0.6)

        # Voting
        vote_boxes = VGroup()
        for i, (col, cls) in enumerate(zip(tree_colors, ["Clase A", "Clase A", "Clase B"])):
            box = RoundedRectangle(width=2.2, height=0.6, corner_radius=0.1,
                                   fill_color=col, fill_opacity=0.8, stroke_width=0)
            box.move_to([(i - 1) * 3.5, -3.8, 0])
            lbl = Text(cls, font_size=14, color=WHITE, font="Arial").move_to(box.get_center())
            vote_boxes.add(VGroup(box, lbl))

        vote_title = Text("Votos individuales", font_size=16, color=GOLD, font="Arial").move_to(DOWN * 3.0)
        self.play(Write(vote_title), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(v) for v in vote_boxes], lag_ratio=0.3), run_time=0.8)
        self.wait(0.5)

        final = RoundedRectangle(width=3.5, height=0.9, corner_radius=0.15,
                                  fill_color=GOLD, fill_opacity=1, stroke_width=0)
        final.move_to(DOWN * 0.0 + RIGHT * 5.0)
        final_lbl = Text("Clase A (2/3)", font_size=18, color="#0d1b2a", font="Arial", weight=BOLD).move_to(final.get_center())
        final_title = Text("Voto mayoritario", font_size=14, color=GOLD, font="Arial").next_to(final, UP, buff=0.2)
        self.play(FadeIn(final), Write(final_lbl), Write(final_title), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  KNN
# ══════════════════════════════════════════════════════════════════════════════
class KNNScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("K-Nearest Neighbors — Clasificacion por Proximidad", font_size=20, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.0},
            x_length=7, y_length=5, tips=False,
        ).shift(LEFT * 0.5)
        self.play(Create(axes), run_time=0.5)

        np.random.seed(7)
        cls0_pts = np.random.multivariate_normal([-1.5, -1.0], [[0.5, 0], [0, 0.5]], 15)
        cls1_pts = np.random.multivariate_normal([1.5,  1.0],  [[0.5, 0], [0, 0.5]], 15)

        dots0 = VGroup(*[Dot(axes.c2p(x, y), radius=0.09, color=BLUE_C) for x, y in cls0_pts])
        dots1 = VGroup(*[Dot(axes.c2p(x, y), radius=0.09, color=RED_C)  for x, y in cls1_pts])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots0], lag_ratio=0.04),
            LaggedStart(*[GrowFromCenter(d) for d in dots1], lag_ratio=0.04),
            run_time=1.0,
        )

        # Punto de consulta
        query = np.array([0.2, 0.3])
        q_dot = Dot(axes.c2p(*query), radius=0.18, color=GOLD)
        q_lbl = Text("?", font_size=24, color=GOLD, font="Arial").next_to(q_dot, UR, buff=0.1)
        self.play(GrowFromCenter(q_dot), Write(q_lbl))
        self.wait(0.5)

        # Panel derecho
        panel_x = RIGHT * 5.5
        k_title = Text("K = 3 vecinos", font_size=17, color=GOLD, font="Arial").move_to(panel_x + UP * 2.5)
        self.play(Write(k_title))

        # Encontrar los 3 más cercanos
        all_pts = list(cls0_pts) + list(cls1_pts)
        all_cls = [BLUE_C] * 15 + [RED_C] * 15
        dists = [np.linalg.norm(p - query) for p in all_pts]
        top3_idx = np.argsort(dists)[:3]

        circles = VGroup()
        lines = VGroup()
        log_entries = []
        for rank, idx in enumerate(top3_idx):
            pt = all_pts[idx]
            col = all_cls[idx]
            cls_name = "Azul" if col == BLUE_C else "Rojo"
            c = Circle(radius=0.22, color=GOLD, stroke_width=2).move_to(axes.c2p(*pt))
            ln = Line(axes.c2p(*query), axes.c2p(*pt), color=GOLD, stroke_width=1.2, stroke_opacity=0.7)
            d_lbl = Text(f"Vecino {rank+1}: {cls_name}  d={dists[idx]:.2f}",
                         font_size=13, color=col, font="Arial").move_to(panel_x + UP * (1.8 - rank * 0.55))
            circles.add(c)
            lines.add(ln)
            log_entries.append(d_lbl)
            self.play(Create(c), Create(ln), FadeIn(d_lbl), run_time=0.5)
            self.wait(0.3)

        # Decisión
        result_lbl = Text("Mayoria: Azul (2/3)", font_size=16, color=BLUE_C, font="Arial", weight=BOLD)\
            .move_to(panel_x + DOWN * 0.5)
        result_box = SurroundingRectangle(result_lbl, color=BLUE_C, buff=0.12, corner_radius=0.1)
        self.play(Write(result_lbl), Create(result_box), run_time=0.6)
        self.play(q_dot.animate.set_color(BLUE_C), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════
class GradientBoostingScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Gradient Boosting — Correccion Secuencial", font_size=21, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        np.random.seed(0)
        xs = np.linspace(-3, 3, 20)
        ys = np.sin(xs) + np.random.randn(20) * 0.3

        axes = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[-2, 2, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.2},
            x_length=8, y_length=4.5, tips=False,
        ).shift(DOWN * 0.3)
        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.08, color=NAVY_LIGHT) for x, y in zip(xs, ys)])
        self.play(Create(axes), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04), run_time=0.8)

        # Panel izquierdo
        panel_x = LEFT * 5.5
        log_title = Text("Iteraciones", font_size=16, color=GOLD, font="Arial").move_to(panel_x + UP * 2.8)
        sep = Line(panel_x + UP * 2.5 + LEFT * 1.2, panel_x + UP * 2.5 + RIGHT * 1.2,
                   color=NAVY_LIGHT, stroke_width=1)
        self.play(FadeIn(log_title), Create(sep))

        predictions = np.zeros(20)
        colors_iter = [RED_C, BLUE_C, GREEN_C, GOLD]
        entries = []

        for it in range(4):
            residuals = ys - predictions
            degree = min(it + 1, 3)
            coeffs = np.polyfit(xs, residuals, degree)
            correction = np.polyval(coeffs, xs)
            predictions = predictions + 0.5 * correction

            pred_line = axes.plot(lambda x, c=np.polyfit(xs, predictions, 5): np.polyval(c, x),
                                  color=colors_iter[it], stroke_width=2.5, x_range=[-3, 3])

            entry = Text(f"It.{it+1}  Error={np.mean(residuals**2):.3f}",
                         font_size=13, color=colors_iter[it], font="Arial")\
                .move_to(panel_x + UP * (2.1 - it * 0.52))
            entries.append(entry)
            self.play(Create(pred_line), FadeIn(entry), run_time=0.7)
            self.wait(0.6)

        done = Text("Convergencia — error minimo", font_size=16, color=GREEN_C, font="Arial")\
            .move_to(panel_x + DOWN * 0.3)
        self.play(FadeIn(done), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
class NaiveBayesScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Naive Bayes — Teorema de Bayes", font_size=22, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-5, 5, 1], y_range=[-0.05, 0.55, 0.1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.2},
            x_length=9, y_length=4.5, tips=False,
        ).shift(DOWN * 0.5)
        self.play(Create(axes), run_time=0.5)

        # Gaussian distributions for each class
        def gauss(x, mu, sigma): return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

        mu0, sigma0 = -1.5, 1.0
        mu1, sigma1 =  1.5, 1.0

        curve0 = axes.plot(lambda x: gauss(x, mu0, sigma0), color=BLUE_C, stroke_width=2.5)
        curve1 = axes.plot(lambda x: gauss(x, mu1, sigma1), color=RED_C,  stroke_width=2.5)

        area0 = axes.get_area(curve0, x_range=[-5, 0], color=BLUE_C, opacity=0.2)
        area1 = axes.get_area(curve1, x_range=[0, 5],  color=RED_C,  opacity=0.2)

        lbl0 = Text("P(x | Clase 0)", font_size=16, color=BLUE_C, font="Arial").move_to(axes.c2p(-2.8, 0.48))
        lbl1 = Text("P(x | Clase 1)", font_size=16, color=RED_C,  font="Arial").move_to(axes.c2p( 2.8, 0.48))

        self.play(Create(curve0), Create(curve1), run_time=0.8)
        self.play(FadeIn(area0), FadeIn(area1), Write(lbl0), Write(lbl1), run_time=0.6)

        threshold = DashedLine(axes.c2p(0, 0), axes.c2p(0, 0.52),
                                color=GOLD, stroke_width=2, dash_length=0.15)
        thr_lbl = Text("Frontera de\ndecision", font_size=14, color=GOLD, font="Arial")\
            .move_to(axes.c2p(1.2, 0.43))
        self.play(Create(threshold), Write(thr_lbl), run_time=0.6)

        # Bayes formula panel
        panel = VGroup(
            Text("Teorema de Bayes", font_size=15, color=GOLD, weight=BOLD, font="Arial"),
            Text("P(y|x) = P(x|y) * P(y) / P(x)", font_size=13, color=WHITE, font="Arial"),
            Text("Supuesto: features independientes", font_size=12, color=NAVY_LIGHT, font="Arial"),
        ).arrange(DOWN, buff=0.25).to_corner(UR).shift(DOWN * 0.3)
        self.play(FadeIn(panel), run_time=0.6)

        # Moving query point
        x_t = ValueTracker(-4)
        q_dot = always_redraw(lambda: Dot(
            axes.c2p(x_t.get_value(), gauss(x_t.get_value(), mu0, sigma0)),
            color=GOLD, radius=0.13
        ))
        pred_text = always_redraw(lambda: Text(
            "Clase 0" if x_t.get_value() < 0 else "Clase 1",
            font_size=18, color=BLUE_C if x_t.get_value() < 0 else RED_C, font="Arial"
        ).move_to(LEFT * 5.0 + DOWN * 3.0))

        self.play(FadeIn(q_dot), FadeIn(pred_text))
        self.play(x_t.animate.set_value(4), run_time=3.0, rate_func=smooth)
        self.wait(1.0)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
class DBSCANScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("DBSCAN — Clustering por Densidad", font_size=22, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            axis_config={"color": NAVY_LIGHT, "stroke_width": 1.0},
            x_length=6.5, y_length=5.5, tips=False,
        ).shift(RIGHT * 1.5)
        self.play(Create(axes), run_time=0.5)

        np.random.seed(12)
        cluster_A = np.random.multivariate_normal([-2, 1.5], [[0.25, 0], [0, 0.25]], 18)
        cluster_B = np.random.multivariate_normal([1.5, -1.5], [[0.3, 0.1], [0.1, 0.3]], 18)
        outliers  = np.array([[-3.5, -2.5], [3.5, 2.5], [0.5, 2.8]])

        all_pts_white = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.08, color=WHITE)
            for x, y in list(cluster_A) + list(cluster_B) + list(outliers)
        ])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in all_pts_white], lag_ratio=0.03), run_time=0.9)

        # Panel izquierdo
        panel_x = LEFT * 4.5
        p_title = Text("Algoritmo DBSCAN", font_size=16, color=GOLD, font="Arial").move_to(panel_x + UP * 2.8)
        sep = Line(panel_x + UP * 2.5 + LEFT * 1.3, panel_x + UP * 2.5 + RIGHT * 1.3,
                   color=NAVY_LIGHT, stroke_width=1)
        self.play(FadeIn(p_title), Create(sep))

        steps = [
            ("1. Radio eps = 0.8", WHITE),
            ("2. min_samples = 3", WHITE),
            ("3. Puntos nucleo:", GOLD),
            ("   alta densidad", GOLD),
            ("4. Expandir clusters", BLUE_C),
            ("5. Outliers = ruido", RED_C),
        ]
        step_objs = []
        for i, (txt, col) in enumerate(steps):
            s = Text(txt, font_size=13, color=col, font="Arial")\
                .move_to(panel_x + UP * (2.0 - i * 0.48))
            step_objs.append(s)

        # Show eps circle on a core point
        core_pt = axes.c2p(*cluster_A[8])
        eps_circle = Circle(radius=axes.get_x_axis().get_unit_size() * 0.8,
                            color=GOLD, stroke_width=1.5, stroke_opacity=0.6).move_to(core_pt)

        for i, s in enumerate(step_objs[:2]):
            self.play(FadeIn(s), run_time=0.4)
            self.wait(0.3)

        self.play(Create(eps_circle), FadeIn(step_objs[2]), FadeIn(step_objs[3]), run_time=0.6)
        self.wait(0.5)

        # Color clusters
        dots_A = VGroup(*[Dot(axes.c2p(x, y), radius=0.08, color=BLUE_C) for x, y in cluster_A])
        dots_B = VGroup(*[Dot(axes.c2p(x, y), radius=0.08, color=GREEN_C) for x, y in cluster_B])
        dots_out = VGroup(*[Dot(axes.c2p(x, y), radius=0.10, color=RED_C) for x, y in outliers])

        self.play(FadeIn(step_objs[4]), run_time=0.3)
        self.play(Transform(all_pts_white[:18], dots_A), run_time=0.7)
        self.play(Transform(all_pts_white[18:36], dots_B), run_time=0.7)
        self.wait(0.4)
        self.play(FadeIn(step_objs[5]), run_time=0.3)
        self.play(Transform(all_pts_white[36:], dots_out), run_time=0.5)

        # Labels
        lbl_a = Text("Cluster A", font_size=14, color=BLUE_C, font="Arial").move_to(axes.c2p(-2.5, 2.5))
        lbl_b = Text("Cluster B", font_size=14, color=GREEN_C, font="Arial").move_to(axes.c2p(2.0, -2.3))
        lbl_out = Text("Outlier", font_size=13, color=RED_C, font="Arial").move_to(axes.c2p(3.5, 2.3))
        self.play(Write(lbl_a), Write(lbl_b), Write(lbl_out), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
class HierarchicalScene(Scene):
    def construct(self):
        self.camera.background_color = SOFT_BG

        title = Text("Clustering Jerarquico — Dendrograma", font_size=21, color=WHITE, font="Arial").to_edge(UP)
        self.play(Write(title))

        # Puntos de datos
        pts = np.array([[-3, -1], [-2.5, -0.5], [-2, -1.5],
                        [1, 1.5], [1.5, 0.8], [0.8, 1.2],
                        [3, -1.5], [2.5, -1], [3.5, -0.8]])
        colors_final = [BLUE_C]*3 + [GREEN_C]*3 + [RED_C]*3

        # Escala de puntos al espacio de pantalla
        scale = 1.0
        screen_pts = [np.array([p[0]*scale, p[1]*scale, 0]) for p in pts]

        dots = VGroup(*[Dot(pos, radius=0.10, color=WHITE) for pos in screen_pts])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.07), run_time=0.8)

        # Panel derecho: dendrograma simplificado
        panel_x = RIGHT * 4.5
        dend_title = Text("Dendrograma", font_size=17, color=GOLD, font="Arial").move_to(panel_x + UP * 3.0)
        self.play(Write(dend_title))

        leaf_y = -2.8
        node_positions = {
            i: np.array([panel_x[0] - 1.5 + i * 0.42, leaf_y, 0])
            for i in range(9)
        }
        leaf_dots = VGroup(*[
            Dot(node_positions[i], radius=0.07, color=colors_final[i])
            for i in range(9)
        ])
        self.play(FadeIn(leaf_dots), run_time=0.5)

        # Fases de fusión
        merges = [
            ([0, 1], -1.8, BLUE_C),
            ([6, 7], -1.8, RED_C),
            ([3, 4], -1.8, GREEN_C),
            ([0, 2], -0.8, BLUE_C),  # merge del sub-cluster ya formado con p2
            ([5, 3], -0.8, GREEN_C),
            ([6, 8], -0.8, RED_C),
            ([0, 3], 0.5, GOLD),
            ([0, 6], 1.8, GOLD),
        ]

        existing = {i: node_positions[i] for i in range(9)}
        merge_id = 9
        for (a, b), height, col in merges:
            if a in existing and b in existing:
                pa = existing[a]
                pb = existing[b]
                mid = np.array([(pa[0]+pb[0])/2, height, 0])
                la = Line(np.array([pa[0], height, 0]), np.array([pa[0], pa[1], 0]),
                          color=col, stroke_width=1.8)
                lb = Line(np.array([pb[0], height, 0]), np.array([pb[0], pb[1], 0]),
                          color=col, stroke_width=1.8)
                horiz = Line(np.array([pa[0], height, 0]), np.array([pb[0], height, 0]),
                              color=col, stroke_width=1.8)
                self.play(Create(la), Create(lb), Create(horiz), run_time=0.4)
                existing[merge_id] = mid
                existing.pop(a)
                existing.pop(b)
                merge_id += 1

        # Color final de los puntos
        colored = VGroup(*[Dot(screen_pts[i], radius=0.10, color=colors_final[i]) for i in range(9)])
        self.play(Transform(dots, colored), run_time=0.8)

        done = Text("3 clusters identificados", font_size=16, color=GREEN_C, font="Arial").to_edge(DOWN, buff=0.4)
        self.play(FadeIn(done), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects), run_time=0.8)
