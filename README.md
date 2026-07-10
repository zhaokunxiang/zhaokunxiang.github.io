# Zhao Kunxiang Academic Homepage

Native Hugo scaffold with no theme and no JavaScript framework.

## Dependencies

- Hugo Extended 0.164.0 or newer
- Git

No Node.js, npm packages, theme, or frontend framework is required. LaTeX is
rendered at build time with Hugo's embedded KaTeX engine and emitted as MathML.

## Local development

```bash
hugo server --poll 700ms --noHTTPCache
```

The project lives on a shared volume, so polling is required for reliable live
reload when Markdown files change.

- English: <http://localhost:1313/>
- 中文: <http://localhost:1313/zh/>

## Production build

```bash
hugo --gc --minify
```

Generated files are written to `public/`.

## Project structure

- `content/en/`, `content/zh/`: bilingual Markdown content
- `layouts/`: handwritten Hugo templates
- `layouts/_markup/render-passthrough.html`: build-time LaTeX rendering
- `assets/css/main.css`: intentionally empty design entrypoint
- `.github/workflows/hugo.yaml`: GitHub Pages build and deployment

The GitHub Pages workflow overrides the local `baseURL` with the deployed URL.
