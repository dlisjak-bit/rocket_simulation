from cProfile import run
from manimlib import *
import numpy as np
import math
from multiprocessing import pool

V_0 = 0.0005  # volumen prazne plastenke
# v_z = v_0 * 0.5 # volumen zraka v plastenki
RHO_V = 1000  # gostota vode
G = 9.81  # gravitacijski pospešek
P_0 = 100000  # zunanji tlak
P_Z = 2.74*P_0 # tlak v plastenki
S_FL = 0.0036  # presek flaše
S_ZAM = 0.00048  # presek zamaška (šobe)
M = 0.03  # masa prazne plastenke
C = 0.35  # koeficient upora
RHO_Z = 1.2  # gostota zraka
Stevilo = 9  # stevilo bunkic


class Raketa:
    """vse funkcionalnosti rakete so zaobjete v tem razredu"""

    def __init__(self, volumen_plastenke, volumen_vode_v_plastenki, gostota_vode,
                 gravitacijski_pospesek, zunanji_tlak, tlak_zraka_v_plastenki, presek_plastenke,
                 presek_zamaska, masa_prazne_plasteke, koeficient_upora, gostota_zraka, zacetni_kot,
                 zacetna_hitrost, zacetni_x, zacetni_y, delta_t):
        self.v_p = volumen_plastenke
        self.v_v = volumen_vode_v_plastenki
        self.rho_v = gostota_vode
        self.g_z = gravitacijski_pospesek
        self.p_0 = zunanji_tlak
        self.p_z = tlak_zraka_v_plastenki
        self.s_fl = presek_plastenke
        self.s_zam = presek_zamaska
        self.m_r = masa_prazne_plasteke
        self.c_u = koeficient_upora
        self.rho_z = gostota_zraka
        self.phi_0 = zacetni_kot * math.pi / 180
        self.v_x = zacetna_hitrost*math.cos(self.phi_0)
        self.v_y = zacetna_hitrost*math.sin(self.phi_0)
        self.l_x = zacetni_x
        self.l_y = zacetni_y
        self.h_v = 0
        self.dh_dt = 0
        self.d_t = delta_t
        self.t_0 = 0
        self.a_x = 0
        self.a_y = 0
        self.izpraznjena = self.h_v > self.v_v / self.s_fl
        self.iniciacija()

    def bernoulli(self):
        """funkcija updata hitrost nižanja gladine"""
        if self.izpraznjena:
            self.dh_dt = 0
            return
        f_1 = self.phi_0
        vol_zraka = self.v_p - self.v_v
        p_1 = self.p_z * ((vol_zraka) / (vol_zraka +
                          self.s_fl * self.h_v))**1.4
        p_2 = self.p_0
        pos = -1 * self.g_z * \
            math.sin(f_1) + self.a_x * math.cos(f_1) + self.a_y * math.sin(f_1)
        p_3 = self.rho_v * pos * (self.v_v / self.s_fl - self.h_v)
        dh_dt_kvadrat = 2 / \
            (self.rho_v * ((self.s_fl / self.s_zam) ** 2) - 1) * (p_1 - p_2 + p_3)
        if dh_dt_kvadrat > 0:
            self.dh_dt = math.sqrt(dh_dt_kvadrat)
        else:
            self.dh_dt = 0

    def newton(self):
        """funkcija updata pospešek v obeh smereh"""
        masa = self.m_r + self.rho_v * self.v_v - self.rho_v * self.s_fl * self.h_v
        vodni_potisk = self.rho_v * self.s_fl ** 2 / masa / self.s_zam * self.dh_dt ** 2
        sila_upora = self.c_u * self.s_fl * self.rho_z * \
            (self.v_x ** 2 + self.v_y ** 2) / 2 / masa
        self.a_x = (vodni_potisk - sila_upora) * math.cos(self.phi_0)
        self.a_y = -self.g_z + \
            (vodni_potisk * int(not self.izpraznjena) -
             sila_upora) * math.sin(self.phi_0)

    def iniciacija(self):
        """izračuna začetne pospeške in hitrost padanja gladine vode"""
        for o_1 in range(10):
            self.bernoulli()
            self.newton()
            o_1 += 1

    def update(self) -> list:
        """naredi majhen korak v času in poračuna vse spremembe"""
        self.bernoulli()
        self.newton()
        self.v_x += self.a_x * self.d_t
        self.l_x += self.v_x * self.d_t
        self.v_y += self.a_y * self.d_t
        self.l_y += self.v_y * self.d_t
        self.h_v += self.dh_dt * self.d_t
        self.phi_0 = math.atan2(self.v_y, self.v_x)
        self.t_0 += self.d_t
        self.izpraznjena = self.h_v > self.v_v / self.s_fl
        return [self.t_0, self.l_x, self.l_y, self.v_x, self.v_y, self.phi_0]


out_all = []
#cas_leta = []

for i in range(1, Stevilo+1):
    moja_raketa = Raketa(V_0, V_0/2, RHO_V, G, P_0, P_Z,
                         S_FL, S_ZAM, M, C, RHO_Z, 10*i, 0, 0, 0, 0.01)
    out = np.empty((0, 6))
    for i in range(1000):
        out = np.vstack((out, moja_raketa.update()))
        if out[-1, 2] < 0:
            #cas_leta[i-1].append((out[-1,0]))
            break
    out_all.append(out)


class Pikica(Scene):
    def construct(self):
        axes = Axes(
            x_range=(0, 30, 10), # os x
            y_range=(0, 20, 5),  # os y
            height=6,
            width=10, # frame videa
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            }, # osi
            y_axis_config={
                "include_tip": False,
            }
        )
        axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=1,
        ) # oznake osi
        self.add(axes) # generiraj (dodaj objektu - sceni)
        dot_array = [] # array objektov raketic 
        potka_array = [] # array objektov potke vsake raketice
        for j in range(Stevilo):
            empty_set = [] # vsaka raketa ima array z objekti potke
            potka_array.append(empty_set)
            for i in range(len(out_all[j][:,2])):
                dot = Dot(color=WHITE, radius=0.0008)
                potka_array[j].append(dot)
        for i in range(Stevilo):
            dot = Dot(color=RED)
            dot.move_to(axes.c2p(0, 0))
            self.play(FadeIn(dot, scale=0.5, run_time=0)) # pokazi rakete
            print(f"fadein {i}")
            dot_array.append(dot) # daj raketo v array, iz tam se premikajo
        tex1 = Tex(r"V_0=" + str(V_0) + r", S=" + str(S_FL)  + r", m_0=" + str(M) , font_size=20)
        tex2 = Tex(r"F_{upor}=\frac{1}{2} CS \rho V^2", font_size=20)
        tex3 = Tex(r"C=" + str(C), font_size=20)
        self.add(VGroup(tex1, tex2, tex3).arrange(DOWN))
        for i in range(1000):
            animations = [] # v vsakem frameu je to array premikov vseh raketic
            animations_potka = [] # ppremik vseh potk za en frame
            for j in range(Stevilo):
                if i<len(out_all[j][:,2]):
                    animations.append(ApplyMethod(dot_array[j].move_to, (axes.c2p(out_all[j][:, 1][i], out_all[j][:, 2][i]))))
                    animations_potka.append(ApplyMethod(potka_array[j][i].move_to, (axes.c2p(out_all[j][:, 1][i], out_all[j][:, 2][i]))))
            self.play(*animations, run_time=0.01) 
            self.play(*animations_potka, run_time=0) # prikazi animacijo enega frama, vse se premakne naenkrat
