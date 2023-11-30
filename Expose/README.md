# LaTeX Thesis-Vorlage FIW

Vorlage für Abschlussarbeiten der FIW mit dem Textsatzsystem [LaTeX](https://www.latex-project.org/).

## Inhalt

```
anlagen/                           Anlagen, Anhänge, etc.
    beispiel.tex                   Beispielanlage
    cd.tex                         Dateibaum der CD, die zur gedruckten Version gehört
    exampleCode.c                  Beispiel-Programmcode
bilder/                            Grafiken
    HSLogo.jpg                     Beispielgrafik
kapitel/                           Verzeichnis für Kapitel
    aufgabenstellung.tex           Aufgabenstellung
    beispiel.tex                   Beispielkapitel
    einleitung.tex                 Platzhalter Einleitung
    schluss.tex                    Platzhalter Schluss
tools/                             Kleine Hilfsprogramme
    dirtree.pl                     Perl Skript zur semi-automatischen Erzeugung von DirTree Inhalten
verzeichnisse/                     Abkürzungen, Symbole, Glossar
    abkuerzungen.tex               Inhalt Abkürzungsverzeichnis
    glossar.tex                    Inhalt Glossar
    symbole.tex                    Inhalt Symbolverzeichnis
fiwthesis.cls                      Dokumentenklasse (nur editieren, wenn Sie wissen was sie tun!)
Makefile                           Make-Datei zum Erstellen der Arbeit
quellen.bib                        Literaturdatenbank
README.md                          Informationen zur Vorlage (diese Datei)
thesis.tex                         Zentraldokument / Grundgerüst der Arbeit
.drone.yml                         Konfigurationsdatei für Drone CI/CD Pipeline
```

## Installation der LaTeX-Umgebung

Das Template bietet die Möglichkeit, verschiedene Betriebssysteme und Entwicklungsumgebungen zur Erstellung der Arbeit zu nutzen.
Mehr Informationen entnehmen Sie bitte den folgenden Installationsanweisungen.

### Voraussetzungen

Zur Erstellung Ihrer Abschlussarbeit im PDF-Format benötigen Sie:

- eine aktuelle LaTeX-Distribution
  ([TeX Live](https://www.tug.org/texlive/), [MiKTeX](https://miktex.org/) oder [MacTeX](https://www.tug.org/mactex/))
- den LaTeX-Compiler `lualatex` (üblicherweise in den Distributionen enthalten)
- [Python](https://www.python.org/downloads/) (Version 2.6 oder höher) für das `minted` LaTeX-Paket
- [Perl](https://www.perl.org/get.html) für Hilfsprogramme
- die Perl-Skripte [`makeglossaries`](https://ctan.org/tex-archive/macros/latex/contrib/glossaries) und [`biber`](https://biblatex-biber.sourceforge.net/) zum Erstellen des Glossars sowie des Abkürzungs- und Symbolverzeichnisses und der Bibliographie (sollten ebenfalls in der Distribution enthalten sein)

Aufgrund der Nutzung der Pakete _fontspec_ und _selnolig_ muss zwangsläufig
LuaLaTeX als Compiler verwendet werden.

Sollten Sie einen speziellen LaTeX-Editor wie Kile, TeXstudio, TeXnicCenter
o. Ä. verwenden, sind die erforderlichen Abhängigkeiten i.d.R. schon erfüllt.
Prüfen Sie aber dennoch, ob die Pfade zu den erforderlichen Kommandos,
insbesondere `makeglossaries`, korrekt gesetzt sind.

### GNU/Linux

#### Ubuntu / Debian / Linux Mint / ElementaryOS / Pop!\_OS / Raspberry Pi OS

1. Installieren Sie die erforderlichen Pakete:

```sh
sudo apt-get install texlive-base texlive-xetex texlive-luatex texlive-science texlive-latex-extra texlive-bibtex-extra texlive-lang-german texlive-pictures biber python3 python3-pygments ttf-bitstream-vera
```

### MacOS

1. [Laden Sie MacTeX herunter](https://tug.org/mactex/mactex-download.html).
1. Folgen Sie den Installationsanweisungen.

### Windows 10 / Windows 11

Die empfohlene Vorgehensweise für Windows 10 und 11 ist das Verwenden des Windows Subsystem for Linux (WSL). Für die Bearbeitung empfiehlt sich VSCode mit der WSL-Erweiterung.

1. Installieren der [Erweiterung 'WSL'](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl).
1. Verbinden zu WSL via `ctrl+shift+p` und Ausführen von `>WSL: connect to WSL` oder per Click auf den Remote Window Button ganz unten links in der Ecke des Fensters. Standardmäßig wird dadurch Ubuntu 22.04 installiert.
1. Über die Menüleiste `Terminal/Neues Terminal` ein Terminal in der WSL öffnen und folgenden Befehl ausführen:

   ```sh
   sudo apt-get update && sudo apt-get upgrade && sudo apt-get install texlive-base texlive-xetex texlive-luatex texlive-science texlive-latex-extra texlive-bibtex-extra texlive-lang-german texlive-pictures biber python3 python3-pygments ttf-bitstream-vera
   ```

    Dies führt eine Systemaktualisierung durch und installiert alle nötigen Pakete.

## Erstellung

Die Reihenfolge der Kommandos zum Erstellen einer PDF-Version der Thesis
ist identisch zu der Regel 'all' in der Make-Datei:

1. `lualatex thesis`
1. `biber thesis`
1. `makeglossaries thesis`
1. `lualatex thesis`
1. `lualatex thesis`
1. `lualatex thesis`

oder einfach nur `make`, falls installiert.

Bitte achten Sie darauf, dass die "Build Chain" Ihres LaTeX-Editors
genau so aufgebaut ist, falls Sie kein Makefile benutzen.

## Editor-Integration

### Visual Studio Code

#### Installation

1. Installieren Sie eine aktuelle LaTeX-Distribution, z. B. [TeX Live](https://www.tug.org/texlive/).
2. Öffnen Sie die Vorlage in VS Code (Datei > Ordner öffnen).
3. VS Code fordert Sie nun auf, die LaTeX-Erweiterung zu installieren.
   Ist dies nicht der Fall, führen Sie bitte eine manuelle Installation der LaTeX-Erweiterung aus, indem Sie im Extension-Explorer nach [_LaTeX Workshop_](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) suchen und diese installieren.

#### Nutzung

Im Ordner `.vscode` sind die zwei Dateien `extensions.json` und `settings.json` hinterlegt, welche empfohlene Erweiterungen und Einstellungen für die Erweiterung `LaTeX-Workshop` enthalten.

Die Erweiterung `LaTeX-Workshop` ist ein mächtiges Toolkit, welches für das Schreiben in VSCode sehr zu empfehlen ist. Es bietet Code Snippets, Autovervollständigung von Kommandos und Environments. Für weitere optionale nützliche Funktionen wie `latexindent` und `synctex` werden jedoch Pakete benötigt, welche manuell nachinstalliert und konfiguriert werden müssen. Mehr Informationen über die Konfiguration von _LaTeX Workshop_ finden Sie im [Projektwiki](https://github.com/James-Yu/LaTeX-Workshop/wiki) auf Github.

Standardmäßig werden alle erzeugten Log-Dateien im File-Explorer ausgeblendet, um eine bessere Übersicht zu gewährleisten.

## Hinweis für Overleaf-Nutzer

Sollten Sie diese Vorlage in Overleaf verwenden, so beachten Sie bitte,
dass die Datei 'thesis.tex' als Zentraldokument eingestellt ist (Projekteinstellungen).

## Dieses Repository als Vorlage für eigenes Repository nutzen

Bei der Erstellung Ihres eigenen Repository für die zu schreibende Arbeit kann `eteich/fiw-thesis` als Vorlage ausgewählt werden. Damit erhalten Sie die momentan aktuelle Version automatisch als Grundlage ihrer Arbeit.

## Anpassung

Beginnen Sie mit der Anpassung der Arbeit, indem Sie Ihre persönlichen
Angaben in der Datei 'thesis.tex' anstelle der Platzhaltertexte einfügen.

Im Verzeichnis 'kapitel/' befindet sich bereits ein Beispielkapitel. Wenn
Sie neu in Sachen LaTeX sind, schauen Sie sich dieses an.

Erstellen Sie für jedes Kapitel eine eigene Datei im Verzeichnis 'kapitel/'
und referenzieren Sie diese Datei in 'thesis.tex' nach der Einleitung.
Auf diese Weise können Sie die Gliederung schnell ändern.

Diese Vorlage bindet eine kleine Auswahl nützlicher Pakete ein.
Sollten Sie weitere Pakete benötigen, binden Sie diese in der Präambel von
'thesis.tex' ein. Alle Funktionen jedes Pakets zu beschreiben ist an dieser
Stelle nicht sinnvoll. Alle verwendeten Pakete sind im [CTAN](https://ctan.org/)
zu finden, genau wie die zugehörigen Dokumentationen, welche einen tieferen Einblick
in die jeweiligen Pakete liefern.

Anlagen werden der klaren Trennung halber in dem separaten Verzeichnis
'anlagen/' gespeichert; Bilder und Grafiken legen Sie bitte im Verzeichnis
'bilder' ab.

Die Datei 'fiwthesis.cls' beinhaltet alle Formatierungsoptionen sowie die
Titelseite der Arbeit und die Selbstständigkeitserklärung. Bitte ändern Sie
den Text dieser nur, wenn es sich dabei um eine offizielle Änderung handelt.
In diesem Fall veröffentlichen Sie bitte diese Änderung im [Git](https://giw.fiw.hs-wismar.de).

Wenn Sie Abkürzungs-, Symbolverzeichnis und ein Glossar anlegen, muss das
Programm `makeglossaries` unbedingt auf Ihrem Computer installiert sein.
Die nötigen Dateien für Abkürzungsverzeichnis und Glossar werden i.d.R. nicht
automatisch vom LaTeX-Compiler erstellt.

## Drone CI/CD Pipeline

Für dieses Projekt existiert eine [Drone CI/CD](https://www.drone.io) Pipeline welche in der Datei `.drone.yml` definiert ist. Diese Datei dient nur zur Automatisierung der Weiterentwicklung des Projekts und ist für das Bauen des LaTeX-Projekts nicht erforderlich.

Die Pipeline besteht aus zwei Schritten: 'make project' und 'pdf push master'. Ersterer wird bei jedem Push ausgeführt und versucht das Projekt mittels `make clean && make all` von Grund auf neu zu bauen. Der zweite Schritt wird nur ausgeführt, wenn der erste Schritt erfolgreich beendet wurde UND als aktueller Branch `master` ausgewählt ist. Für die erzeugte PDF-Datei wird dann ein Commit erstellt und auf den `master` Branch gepusht.

## Mitarbeit

Dieses Projekt ist zugänglich im [Git der FIW](https://git.fiw.hs-wismar.de/eteich/fiw-thesis).

Möchten Sie Änderungen zum Projekt beitragen, eröffnen Sie bitte einen [Pull-Request](https://git.fiw.hs-wismar.de/eteich/fiw-thesis/pulls) (nur für den Branch `development`).

## Fragen

Fragen, Probleme oder Fehler können Sie im Git-Repository als [Issues](https://git.fiw.hs-wismar.de/eteich/fiw-thesis/issues) eröffnen.
Das hilft auch zukünftigen Nutzern bei der Beantwortung von Fragen, die eventuell häufiger gestellt werden.

## Verwandte Projekte

### HSW-Beamer

Präsentationsfolien im Design der Hochschule Wismar mit LaTeX.

[https://git.fiw.hs-wismar.de/eteich/hsw-beamer](https://git.fiw.hs-wismar.de/eteich/hsw-beamer)

### FIW-Poster

Poster zur Abschlussarbeit mit LaTeX.

[https://git.fiw.hs-wismar.de/eteich/fiw-poster](https://git.fiw.hs-wismar.de/eteich/fiw-poster)
