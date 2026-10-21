# Постер NeurIPS 2026

Один постер по статье *Truth as a Compression Artifact in Language Model
Training* (main track, poster session, Сидней, 6–12 декабря 2026).
Переписан 2026-10-21 под camera-ready (`../main_v2.tex`): основная мера —
aggregate share, контроль с обучаемым селектором, снятые результаты
(multi-rule, лестница) показаны как снятые. Автор назван, аффилиации нет —
как в статье.

| Файл | Что это |
|---|---|
| `poster.tex` | исходник, 48 × 36" landscape (122 × 91 см), beamerposter |
| `poster.pdf` | готовый PDF |
| `poster_preview.png` | растровое превью (100 dpi) |
| `talking_points.md` | скрипт у постера: 30 сек / 2 мин / 5 мин и ответы на вопросы |
| `logo_salute.png`, `макет.pptx`, `poster.pptx` | исторические материалы внутреннего доклада, к постеру NeurIPS отношения не имеют |

Фигуры берутся из `../`: `figure_mass_control.png` (центральный контраст и
лестница) и `figure_poster_learnable.png` (контроль с обучаемым селектором,
строится `analysis/plot_poster_learnable.py`).

## Сборка

```bash
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd paper_conf/poster
pdflatex poster.tex && pdflatex poster.tex
pdftoppm -r 100 -png -singlefile poster.pdf poster_preview
```

Нужны пакеты `beamer`, `beamerposter`, `pgf`, `type1cm`, `ragged2e`
(поставлены в TinyTeX 2026-10-21).

## Перед печатью

1. Вычитать числа против `../main_v2.tex`: все они оттуда, но после любой
   правки статьи постер надо сверить заново.
2. Проверить размер доски, объявленный для сессии в Сиднее; 48 × 36" —
   стандартный формат NeurIPS, при другом формате менять `beamerposter` в
   преамбуле.
3. Печать: партнёр конференции ARC, <https://neurips.myprintdesk.net/login>,
   заказывать минимум за пять дней; либо любая широкоформатная типография.
