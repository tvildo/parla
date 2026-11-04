# Accessibility

Target: **WCAG 2.2 level AA** for the website (parla.cc, every language) and the extension's popup and settings page.

## Website

- **Keyboard**: everything works without a mouse. A "Skip to content" link is the first Tab stop; focus is always
  visible; the language menu is a native disclosure (`<details>`); the demo's quality is a native select, its strength
  a slider that announces its value.
- **Screen readers**: landmarks (header, navigation, main, footer), headings in order, every control named. The
  demo announces its state ("Microphone on: noise suppression is running", "Stopped", errors) through a status
  region; the noise meter's two layers (original, with Parla) are each `role="meter"` with a value, and its number
  ("16× quieter") is plain text; recording and quality changes are disabled while they can't be used.
- **Languages**: `lang` on every page (21 languages), `dir="rtl"` for Arabic and Urdu with mirrored layout;
  `hreflang` links between translations; the browser's language is offered, never forced.
- **Contrast**: text meets 4.5:1 (large text 3:1) on the dark background.
- **Motion**: animations (the hero wave, the meter's transitions) stop with `prefers-reduced-motion`.

## Extension

- Popup and settings: every control has a label; the strength slider announces its value ("80%"); the status line is
  a polite live region and only announces changes; errors are alerts; focus is visible; `lang`/`dir` follow the
  chosen language (right-to-left included); light and dark themes follow the system.
- The popup's noise meter is visual (`aria-hidden`); its number ("16× quieter", plain text) and the status line carry
  the same information. The settings tabs follow the WAI-ARIA tabs pattern (arrow keys, Home / End, mirrored in
  right-to-left languages).

## How it is tested

`python build.py test` (Docker, Playwright) runs:

- **axe-core** (rule sets wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa) on the landing page (menu closed and open),
  privacy and demo pages (idle and running) in English, Georgian and Arabic, and on the extension's popup and
  settings in English and Arabic: no violations allowed.
- **Keyboard-only** use of the demo: Tab to the skip link and the start button, start with Enter, change quality
  from the keyboard, record with Enter; the status announcement is checked. The extension's popup is also checked with
  its noise meter visible, during a call.

Found a barrier? Please open an issue.
