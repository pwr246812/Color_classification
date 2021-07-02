# define bt 3
# define fr A0
# define fg A1
# define fb A2
 
 
constexpr int k = 3;
constexpr int h = 6;
constexpr int r = 5;
constexpr int dlugosc_ciagu = 25;
 
constexpr float gamma = 0.8;
constexpr int limit_epok = 1000;
 
bool printuj = true;
 
int stan = 0;
int epoki = 0;
 
int min_ciag = 1024;
int max_ciag = 0;
 
float wagi[h+r][h+1] = {
  { -4.363086410566874, 14.352295884960945, -11.658731129739232, 12.356358167423721, 0.585009, 0.479873, 0.350291},
  { 3.89397180944362, 1.8786628499053044, 2.3486960469547458, 1.636580239467388, 0.858943, 0.710501, 0.513535},
  { -8.300818295962918, -17.756835044435213, 32.745365556230176, -10.842647970348628, 0.147313, 0.165899, 0.988525},
  { 4.21279297842502, 1.0389841287596475, 1.9179837882509223, 1.7066426092869482, 0.37788, 0.531663, 0.571184},
  { 3.8576989116113856, 1.3020596162517186, 2.0362148713303476, 2.3087881431251334, 0.450789, 0.352123, 0.0570391},
  { 4.405094546570085, 1.4704597736446032, 2.1942504290886053, 1.7184071099946345, 0.30195, 0.875973, 0.726676},
  { 3.757780769022439, -15.67791969027995, 0.8739879907195517, -16.500958027526252, 1.6846171048517136, 0.7195515417948471, 1.7444464689372592},
  { -6.723352341341053, 20.48470911871675, -3.910758828046242, 11.453047574359708, -3.5971049480087505, -4.120669351833706, -4.739699942124093},
  { -1.548634921135463, 7.465436252639203, 0.2796618077804065, -12.376709911768323, -0.46137864130441836, -0.6885161237742985, -0.21388699711620754},
  { -2.971914658903397, -21.828144553832793, -2.9318797222182904, 19.459097207977283, -2.9251051640221366, -3.053278788534761, -2.8469150422816316},
  { -1.565071713232639, -0.6241898379155411, -0.03084096680057824, 1.979804970422726, -0.07252369436982095, -0.08985084208442387, -0.5774984369130913},
};
 
float s[h+r] = {};
 
 
float ciag_uczacy[dlugosc_ciagu][k+r] = {};
 
int wczytano = 0;
 
 
float f_akt(float x) {
  return 1/(1 + exp(-x));
}
 
void skaluj(int minimum, int maximum) {
  for (int i = 0; i < dlugosc_ciagu; ++i) {
    int temp_min = min(ciag_uczacy[i][0], ciag_uczacy[i][1]);
    int temp_min_1 = min(temp_min, ciag_uczacy[i][2]);
    if (temp_min_1 < min_ciag) min_ciag = temp_min_1;
    int temp_max = max(ciag_uczacy[i][0], ciag_uczacy[i][1]);
    int temp_max_1 = max(temp_max, ciag_uczacy[i][2]);
    if (temp_max_1 > max_ciag) max_ciag = temp_max_1; 
  }
  for (int i = 0; i < dlugosc_ciagu; ++i)
    for (int j = 0; j < k; ++j) ciag_uczacy[i][j] = (static_cast<float>(ciag_uczacy[i][j]) - min_ciag)/( max_ciag- min_ciag) * (maximum - minimum) + minimum;
}
 
 
float skaluj_odczyt(int x, int minimum, int maximum) {
  return (static_cast<float>(x) - min_ciag)/( max_ciag- min_ciag) * (maximum - minimum) + minimum;
}
 
 
float perceptron(float x1, float x2, float x3, float x4, float x5, float x6, int p) {
   return f_akt(wagi[p][0] + wagi[p][1] * x1+ wagi[p][2] * x2 + wagi[p][3] * x3 + wagi[p][4] * x4 + wagi[p][5] * x5 + wagi[p][6] * x6);
}
 
 
void siec(float x1, float x2, float x3) {
  s[0]  = perceptron(x1, x2, x3, 0, 0, 0, 0);
  s[1]  = perceptron(x1, x2, x3, 0, 0, 0, 1);
  s[2]  = perceptron(x1, x2, x3, 0, 0, 0, 2);
  s[3]  = perceptron(x1, x2, x3, 0, 0, 0, 3);
  s[4]  = perceptron(x1, x2, x3, 0, 0, 0, 4);
  s[5]  = perceptron(x1, x2, x3, 0, 0, 0, 5);
  s[6]  = perceptron(s[0], s[1], s[2], s[3], s[4], s[5], 6);
  s[7]  = perceptron(s[0], s[1], s[2], s[3], s[4], s[5], 7);
  s[8]  = perceptron(s[0], s[1], s[2], s[3], s[4], s[5], 8);
  s[9]  = perceptron(s[0], s[1], s[2], s[3], s[4], s[5], 9);
  s[10] = perceptron(s[0], s[1], s[2], s[3], s[4], s[5], 10);
}
 
void uczenie() {
  for (int i = 0; i < dlugosc_ciagu; ++i) {
    siec(ciag_uczacy[i][0],ciag_uczacy[i][1], ciag_uczacy[i][2]);
    for (int j=h; j<h+r; ++j){
      wagi[j][0] -= gamma * (ciag_uczacy[i][j-h+k] - s[j]) * (-1) * s[j] * (1-s[j]) * 1;
      for (int z=1; z<h+1; ++z) {
        wagi[j][z] -= gamma * (ciag_uczacy[i][j-h+k] - s[j]) * (-1) * s[j] * (1-s[j]) * s[z-1];
      }
    }
    for (int j=0; j<h; j++) {
      float suma = 0;
      for (int p=h; p<h+r; ++p) {
        suma += (ciag_uczacy[i][ p - h + k] - s[p]) * (-1) * s[p] * (1 - s[p]) * wagi[p][j + 1];
      }
      wagi[j][0] -= gamma * 2 * suma * s[j] * (1 - s[j]) * 1;
      for (int z=1; z<k+1; ++z) {
        wagi[j][z] -= gamma * 2 *suma * s[j] * (1 - s[j]) * ciag_uczacy[i][z-1];
      }
    }
  }
}
 
 
int policz_blad(bool verb){
  int blad = 0;
  for (int i = 0; i < dlugosc_ciagu; ++i) {
    siec(ciag_uczacy[i][0],ciag_uczacy[i][1], ciag_uczacy[i][2]);
    float temp_max = s[h];
    int idx = 0;
    for (int j=h+1; j<h+r; ++j) {
      if (temp_max < s[j]) {
        temp_max = s[j];
        idx = j-h;
      }
    }
    blad += (idx != (i%r));
    if (verb) {
      Serial.print("d: ");
      Serial.print((i%r)+1);
      Serial.print(", y: ");
      Serial.print(idx+1);
      Serial.print(", błąd: ");
      Serial.println((i%r) != (idx));
    }
  }
  return blad;
}
 
void setup() {
  // put your setup code here, to run once:
  pinMode(bt, INPUT_PULLUP);
  Serial.begin(9600);
  Serial.println("Wprowadź ciąg uczący");
}
 
void loop() {
  // put your main code here, to run repeatedly:
  switch (stan) {
    case 0:
      if (wczytano < dlugosc_ciagu) {
        if (digitalRead(bt) == LOW) {
          ciag_uczacy[wczytano][0] = analogRead(fr);
          ciag_uczacy[wczytano][1] = analogRead(fg);
          ciag_uczacy[wczytano][2] = analogRead(fb);
          ciag_uczacy[wczytano][wczytano%5+k] = 1;
          Serial.print(wczytano+1);
          Serial.print(": ");
          Serial.print(analogRead(fr));
          Serial.print(", ");
          Serial.print(analogRead(fg));
          Serial.print(", ");
          Serial.print(analogRead(fb));
          Serial.print(", ");
          Serial.println((wczytano%5)+1);
          ++wczytano;
          delay(400);
        }
      } else stan++;
      break;
 
    case 1:
      Serial.println("\nUczenie...");
      skaluj(0, 1);
      int blad = 0;
      blad = policz_blad(false);
      Serial.print("Błąd początkowy: ");
      Serial.println(blad);
      do {
        uczenie();
        blad = policz_blad(false);
        ++epoki;
        if (!(epoki%100)) {
          Serial.print(epoki);
          Serial.print(": ");
          Serial.println(blad);
        }
      } while ((blad) && (epoki < limit_epok));
      Serial.println("");
      Serial.println("TESTOWANIE:");
      Serial.print("Błąd po ");
      Serial.print(epoki);
      Serial.print(". epoce"); 
      Serial.print(": ");
      Serial.println(blad);
      Serial.println("");
      blad = policz_blad(true);
      stan++;
      
    case 2:
    while (true){
      if (printuj) {
        Serial.println("WPROWADŹ PRÓBKĘ");
        printuj = false;  
      }
      if (digitalRead(bt) == LOW) {
        int x1 = analogRead(fr);
        int x2 = analogRead(fg);
        int x3 = analogRead(fb);
        float x1_s = skaluj_odczyt(x1,0,1);
        float x2_s = skaluj_odczyt(x2,0,1);
        float x3_s = skaluj_odczyt(x3,0,1);
        siec(x1_s, x2_s, x3_s);
        float temp_max = s[h];
        int idx = 0;
        for (int j=h+1; j<h+r; ++j) {
          if (temp_max < s[j]) {
            temp_max = s[j];
            idx = j-h;
          }
        }
        Serial.print("Pomiary: ");
        Serial.print(analogRead(fr));
        Serial.print(", ");
        Serial.print(analogRead(fg));
        Serial.print(", ");
        Serial.print(analogRead(fb));
        Serial.print(" | ");
        String kolor = "";
        switch (idx) {
          case 0:
            kolor = "Czarny";
            break;
          case 1:
            kolor = "Biały";
            break;
          case 2:
            kolor = "Czerwony";
            break;
          case 3:
            kolor = "Zielony";
            break;
          case 4:
            kolor = "Niebieski";
            break;
        }
        Serial.println(kolor);
        delay(400);
        printuj = true;
      }
    }
      break;
      
    default:
      Serial.println(stan);
      break;
  }
}
