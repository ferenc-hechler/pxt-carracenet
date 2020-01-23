#include "Mat.h"
#include "Vect.h"
#include "NNLayer.h"
#include "NN.h"
#include <stdio.h>



NN::NN(int num_input) {
	this->num_input = num_input;
	this->num_layers = 0;
	this->layers = 0;
}

int NN::getNumLayers() const {
	return num_layers;
}

NNLayer* NN::getLayer(int n) const {
	return layers[n];
}

void NN::addLayer(int num_output) {
	int old_output = getNumOutput();
	if (num_layers == 0) {
		layers = new NNLayer*[1];
	}
	else {
		NNLayer **old_layers = layers;
		layers = new NNLayer*[num_layers+1];
		for (int i=0; i<num_layers; i++) {
			layers[i] = old_layers[i];
		}
		delete old_layers;
	}
	layers[num_layers] = new NNLayer(old_output, num_output);
	num_layers = num_layers + 1;
}


int NN::getNumInput() const {
	return num_input;
}

int NN::getNumOutput() const {
	if (num_layers == 0) {
		return getNumInput();
	}
	return layers[num_layers-1]->getNumOutput();
}

Vect* NN::forwardPropagate(const Vect *input) {
	if (num_layers == 0) {
		Vect *result = new Vect(input);
		return result;
	}
	Vect *result = (Vect *)input;
	Vect* tempResult = 0;
	for (int i=0; i<num_layers; i++) {
		result = layers[i]->forwardPropagate(result);
		if (tempResult != 0) {
			delete tempResult;
		}
		tempResult = result;
	}
	return result;
}

Vect* NN::backwardPropagate(const Vect *expected_output, const Vect *last_output, float learning_rate) {
	Vect *err = new Vect(last_output);
	err->sub(expected_output);
	for (int n=num_layers-1; n>=0; n--) {
		NNLayer *layer = layers[n];
		Vect *err_in = layer->backwardPropagate(err, learning_rate);
		delete err;
		err = err_in;
	}
	return err;
}



void NN::print() const {
	printf("NN(%d)\n", num_layers);
	for (int i=0; i<num_layers; i++) {
		layers[i]->print();
	}
}

static const float car_race_net_weights12[6][10] = {
   {-1.01473932e+00,  2.24403285e-01, -3.78727221e-01,
	-2.55967027e-01, -2.13737237e+00, -1.69397551e+00,
	-1.35168740e+00,  2.60560291e+00,  1.77057196e+00,
	 1.12098331e-01},
   { 1.71437799e-02, -5.96284010e-01, -1.84542426e-01,
	-6.18842903e-03, -3.82185472e-05,  1.35354279e-01,
	-3.02548170e-02,  3.11272043e-02,  7.71586979e-02,
	-5.36296895e-02},
   {-7.01904628e-01, -6.11479586e-01, -1.43005893e-01,
	-2.59446865e-01, -6.95995020e-02,  3.17330164e-01,
	 1.65307021e-01,  1.01339036e-02, -2.53381822e-01,
	-4.52648243e-01},
   { 1.03188227e+00, -2.81994239e-01, -5.70143314e-01,
	 5.17856402e-01, -9.77040958e-01, -7.48908593e-01,
	 2.35082932e+00,  8.33745519e-01,  2.25636527e-01,
	 3.74108098e-01},
   { 1.48480383e-01, -2.93148309e-02,  2.54670901e-02,
	-8.59271148e-01, -3.17555390e-01, -5.93759610e-01,
	 8.22673334e-02, -1.00515880e+00, -1.02079536e+00,
	-2.99923062e-01},
   { 3.71441564e-01, -5.11423548e-01,  9.05958269e-02,
	 4.94822052e-01,  4.34565856e-02,  3.79489158e-01,
	-1.50650143e-01, -6.49969975e-02,  1.59885835e-01,
	 2.85260351e-01}
};


static const int car_race_net_weights23[10][10] = {
   {-0.48252446, -0.31242442,  0.7978847 ,  0.63695254,  0.08571874,
	 0.2352777 ,  0.37626564, -0.86441104,  0.43496355,  0.88131061},
   {-0.16691325, -0.14140342,  0.46395258,  0.08367866,  0.24348544,
	 0.47350828,  0.35017862,  0.00639813, -0.38335753, -0.47969211},
   {-0.49375735, -0.17804221, -0.44820508, -0.22410053, -0.18829577,
	 0.08402317, -0.06507852, -0.45829743, -0.46624136, -0.54335062},
   {-0.63016982, -0.15652387,  0.22608117,  0.23968742,  0.36190894,
	 0.42863618,  0.52902821,  0.25338619,  0.43982645, -0.45749266},
   { 0.33192987, -0.16444801, -0.3135331 , -2.71759169, -0.44460907,
	-0.19129282,  0.64848857,  0.15878613,  0.15820587,  0.74903227},
   { 0.81458235, -0.28810141,  0.67432995, -1.10478309,  0.04855264,
	-0.43887715,  0.96360309,  1.08009135, -0.10330025,  0.95599494},
   {-0.29491954,  0.79345331,  1.43760569, -0.4623732 ,  0.02299648,
	 0.54773604,  0.63238039,  1.69503432, -0.34519927,  1.01083298},
   {-0.29957984, -0.32201857, -2.00522116, -1.42476826, -0.32674229,
	 1.08631061,  0.24313449,  0.3129696 , -0.30883435,  1.39204808},
   { 0.66502339, -0.64330681, -0.67014253,  1.35737054, -0.43747708,
	-0.30700199, -0.66453304, -0.54382173,  0.11199639, -0.46907358},
   {-0.29033566,  0.25525155, -0.07832724, -0.33004161, -0.21573666,
	-0.34961267, -0.48874181, -0.0610558 , -0.28105171,  0.43650486}
};


static const int car_race_net_weights34[10][3] = {
   {-0.98454737, -0.05663108,  0.62307727},
   { 0.75470584, -0.59085766, -0.60059393},
   { 0.49317212, -2.33637768,  1.15584492},
   { 2.68744439, -1.70389356, -1.6441535 },
   {-0.47749261,  0.47406919,  0.29237924},
   { 0.55921877, -1.43440836,  1.23295603},
   {-0.65110343,  1.21736489, -0.84307694},
   { 0.21611596,  1.49432523, -1.69555537},
   {-0.04586901, -0.02932025, -0.2458539 },
   {-0.75448906, -1.73916759,  2.15074643}
};

void NN::init_car_race_net() const {
	set_weights(layers[0]->getWeights(), car_race_net_weights12,  6, 10 );
	set_weights(layers[1]->getWeights(), car_race_net_weights23, 10, 10 );
	set_weights(layers[2]->getWeights(), car_race_net_weights34, 10,  3 );
}


void set_weights(Mat* weights, float* data, rows, cols) {
	for (int r=0; r<rows; r++) {
		for (int c=0; c<cols; c++) {
			weights->set(r, c, *data);
			data++;
		}
	}
}

NN::~NN() {
	for (int i=0; i<num_layers; i++) {
		delete layers[i];
	}
	num_layers = 0;
	delete layers;
	layers = 0;
}

