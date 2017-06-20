# Spracovanie GPS trackov

Pre nas problem trenujeme diskretne HMM s 3 skrytymi stavmi, jeden pre pohyb autom, jeden peso a jeden ked gps stoji.
Pozorovania vyjadrime ako rozdiely po sebe iducich poloh. Konkretne ako euklidovsku vzdialenost po sebe nasledujucich vektorov (latitude, longitude).
Aby sme mali pozorovani konecne vela, zaokruhlime rozdiely na 3 desatinne miesta a vypocitame vysledny pocet roznych pozorovani podla dostupnych dat.

## Oznacovanie dat
Data sme si zobrazili pomocou gpsvisualizer.com a oznacili jednotlive tracky pismenami 'a', 's' a 'p' analogicky ku stavom 'autom', 'stoji', 'peso'.
Oznacene data pouzijeme na vyhodnotenie uspesnosti natrenovaneho modelu.

## Trenovanie a vyhodnotenie modelu
Parametre HMM natrenujeme pomocou Baum-Welchovho algoritmu. Najpravdepodobnejsiu sekvenciu skrytch stavov zratame pomocou Viterbiho algoritmu. Nasledne vyratame accuracy najpravdepodobnejsej sekvencie s pouzitim oznacenych dat.

Lepsie by zrejme bolo natrenovat spojite HMM, kedze nase riesenie skonverguje do nuly.

Accuracy mame vsak aj tak velmi vysoku pokial povieme, ze gps sa stale pohybovala autom. V nasich oznacenych datach je totizto 35790 merani pocas jazdy autom, 878 merani pocas chodze peso a len 460 merani ked gps stala.
