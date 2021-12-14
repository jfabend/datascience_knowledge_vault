
Irgendwo in Gitlab wurde für ein Repo eine Release Pipeline hinterlegt
Diese beschreibt den Publishing Prozess vom Repo in die Artifactory

Ich kann dem Master (oder einem Branch) in Gitlab einen Tag geben
linkes Menü: Repository > Tags

Dann wird das Paket über die Gitlab Pipeline ins Artifactory gepublisht


Dafür muss in der Setup.py aber die gleiche Versionsnummer stehen

Außerdem muss man drauf achten, dass in jedem Package Ordner
eine __init__.py liegt

Zudem muss man darauf achten, dass in den requirements.txt
und in der setup.py die gleichen Dependencies hinterlegt sind


Man kann in der Tag-Ansicht in Gitlab die gepublishten Versionen
auch wieder löschen

Das gepublishte Paket kann installiert werden,
indem man in der Pip config den Artifactory URL hinterlegt
und in die requirements.txt das gepublishte Package mit Tag reinschreibt