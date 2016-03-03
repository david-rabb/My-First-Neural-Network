package neural;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author David
 */
public class Network4 {
  private static final double rate = 0.5;
  
  Layer layer1;
  Layer layer2;
  
  public Network4(int[] numInputs) {
    layer1 = new Layer(numInputs[0], numInputs[1]);
    layer2 = new Layer(numInputs[1], numInputs[2]);
  }
  
  public void train(double[] input, double[] expected) {
    // feed forward
    layer1.feedForward(input, layer2);
    /*
    System.out.print("\nInput:  ");
    for(int i=0; i<input.length; i++) {
      System.out.print(round(input[i])+" ");
    }
    System.out.print("\nOutput1: ");
    for(Neuron n1 : layer1.neurons) {
      System.out.print(round(n1.output)+" ");
    }
    System.out.print("\nOutput2: ");
    for(Neuron n2 : layer2.neurons) {
      System.out.print(round(n2.output)+" ");
    }
    System.out.println();
    */
    // Back propagation
    layer2.computeError(expected);
    layer1.computeHiddenError(layer2);
    
    // update weights
    for(Neuron n1 : layer1.neurons) {
      n1.updateWeights();
    }
    for(Neuron n2 : layer2.neurons) {
      n2.updateWeights();
    }
    /*
    System.out.print("\nOutput: ");
    for(Neuron n2 : layer1) {
      System.out.print(round(n2.output)+" ");
    }
    System.out.print("\nError: ");
    for(Neuron n2 : layer1) {
      System.out.print(round(n2.error)+" ");
    }
    System.out.print("\nOutput: ");
    for(Neuron n2 : layer2) {
      System.out.print(round(n2.output)+" ");
    }
    System.out.print("\nError: ");
    for(Neuron n2 : layer2) {
      System.out.print(round(n2.error)+" ");
    }
    System.out.println("\nExpected: "+expected);
    */
  }
  
  public void print() {
    System.out.println("Weights: ");
    for(int i=0; i<layer1.neurons.size(); i++) {
      Neuron n = layer1.neurons.get(i);
      System.out.println((i+1)+": "+n);
    }
    for(int i=0; i<layer2.neurons.size(); i++) {
      Neuron n = layer2.neurons.get(i);
      System.out.println((i+1)+": "+n);
    }
    System.out.println();
  }
  
  public static void main(String[] args) {
    double[] i1 = {.05,.1};
    double[] e1 = {.01,.99};
    Network4 t = new Network4(new int[]{2, 2, 2});
    System.arraycopy(new double[]{.15,.2,.35}, 0, t.layer1.neurons.get(0).weight, 0, 3);
    System.arraycopy(new double[]{.25,.3,.35}, 0, t.layer1.neurons.get(1).weight, 0, 3);
    System.arraycopy(new double[]{.40,.45,.6}, 0, t.layer2.neurons.get(0).weight, 0, 3);
    System.arraycopy(new double[]{.50,.55,.6}, 0, t.layer2.neurons.get(1).weight, 0, 3);
    
    t.print();
    for(int i=1; i<10001; i++) {
      t.train(i1, e1);
      System.out.println("#### Round "+i+"  \t"+t.layer2.totalError(i1, e1));
      //t.print();
    }
  }
  
  public double round(double d) {
    return (Math.round(d*1000.0))/1000.0;
  }
  
  class Layer {
    int numInputs;
    double[] input;
    List<Neuron> neurons = new ArrayList<>();
    
    public Layer(int numInputs, int numNeurons) {
      this.numInputs = numInputs;
      for(int i=0; i<numNeurons; i++) {
        Neuron n = new Neuron(this);
        neurons.add(n);
      }
    }
    public void feedForward(double[] _input, Layer next) {
      input = new double[numInputs+1];
      System.arraycopy(_input, 0, input, 0, _input.length);
      input[input.length-1] = 1; // bias
      
      double[] output = new double[neurons.size()];
      for(int i=0; i<neurons.size(); i++) {
        Neuron n = neurons.get(i);
        n.computeOutput();
        output[i] = n.output;
      }
      if (next!=null) next.feedForward(output, null);
    }
    public void computeError(double[] expected) {
      for(int i=0; i<neurons.size(); i++) {
        Neuron n = neurons.get(i);
        n.computeError(expected[i]);
      }
    }
    public void computeHiddenError(Layer next) {
      for(int i=0; i<neurons.size(); i++) {
        Neuron n = neurons.get(i);
        n.computeHiddenError(i, next);
      }
    }
    public double totalError(double[] input, double[] expected) {
      double sum = 0;
      System.out.print("\t\t\t\t\t");
      for(int i=0;i<expected.length; i++) {
        double diff = expected[i] - neurons.get(i).output;
        sum += 0.5*diff*diff;
        System.out.print(neurons.get(i).output+"  ");
      }
      System.out.println();
      return sum;
    }
  }
  class Neuron {
    Layer layer;
    double[] weight;
    double output;
    double[] error; // error for each weight
    double derivative;

    public Neuron(Layer layer) {
      this.layer = layer;
      weight = new double[layer.numInputs+1];
      error = new double[layer.numInputs+1];
      
      // initialize weights randomly
      for(int i=0; i<weight.length; i++) {
        //weight[i] = Math.random();
        weight[i] = 1.0;
      }
    }
    
    public void computeOutput() {
      double sum = 0;
      for(int i=0; i<layer.input.length; i++) {
        sum += weight[i]*layer.input[i];
      }
      output = sigmoid(sum); //sigmoidal activation function
    }

    public void computeError(double expectedValue) {
      derivative = (output - expectedValue) * output * (1.0-output);
      for(int i=0; i<weight.length; i++) {
        error[i] =  derivative * layer.input[i];
        //System.out.println(error[i] + " = " +(output - expectedValue) + ", " + (output * (1.0-output)) + ", "+layer.input[i]);
        //double w = weight[i] - rate*error[i];
        //System.out.println(i+": "+w);
      }
    }
    public void computeHiddenError(int inputNum, Layer next) {
      double dEdO = 0;
      for(int i=0; i<next.neurons.size(); i++) {
        Neuron n = next.neurons.get(i);
        double e = n.derivative * n.weight[inputNum];
        //System.out.println("\te:"+e);
        dEdO += e;
      }
      derivative = dEdO * output * (1.0-output);
      //System.out.println("dEdO:"+dEdO+" \t"+(output * (1.0-output)));
      
      for(int i=0; i<weight.length; i++) {
        error[i] =  derivative * layer.input[i];
        //System.out.println(error[i] + " = " +derivative + ", "+layer.input[i]);
        //double w = weight[i] - rate*error[i];
        //System.out.println(i+": "+w);
      }
    }
    
    public void updateWeights() {
      for(int i=0; i<weight.length; i++) {
        weight[i] -= rate * error[i];
      }
    }
    
    public double sigmoid(double z) {
      return 1.0/(1.0 + Math.exp(-1.0*z));
    }
    
    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      for(int i=0; i<weight.length; i++) {
        sb.append(round(weight[i])).append(" ");
      }
      return sb.toString();
    }
  }
}

