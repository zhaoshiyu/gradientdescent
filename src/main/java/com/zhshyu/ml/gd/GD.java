package com.zhshyu.ml.gd;


/**
 * 梯度下降法在线性回归中的应用
 * 
 * @author Zhao Shiyu
 *
 */
public class GD {
	
	/**
	 * 目标函数
	 * @param theta 参数
	 * @param x 变量
	 * @return 结果
	 * @throws Exception
	 */
	public static double target(double[] theta, double[] x) throws Exception {
		if(theta.length != x.length) throw new Exception("维数不一致！");
		int len = theta.length;
		double ret = 0;
		for(int i = 0; i < len; i++) {
			ret += theta[i] * x[i];
		}
		return ret;
	}
	
	/**
	 * 计算偏差
	 * @param theta 参数
	 * @param x 变量
	 * @param y 真实结果
	 * @return 偏差大小
	 * @throws Exception
	 */
	public static double dev(double[] theta, double[] x, double y) throws Exception {
		if(theta.length != x.length) throw new Exception("维数不一致！");
		return (target(theta, x) - y);
	}
	
	/**
	 * 计算损失值
	 * @param theta 参数集
	 * @param x 变量集
	 * @param y 真实结果集
	 * @return 损失值大小
	 * @throws Exception
	 */
	public static double loss(double[][] theta, double[][] x, double y[]) throws Exception {
		if(theta.length != x.length) throw new Exception("维数不一致！");
		int len = y.length;
		double lossSum = 0;
		for(int i = 0; i < len; i++) {
			lossSum +=  Math.pow(dev(theta[i], x[i], y[i]), 2.0);
		}
		return (lossSum / len);
	}
	
	/**
	 * 更新参数
	 * @param theta 参数
	 * @param x 变量
	 * @param y 真实结果
	 * @param alpha 学习速率
	 * @return 更新后的参数
	 * @throws Exception
	 */
	public static double[] updateTheta(double[] theta, double[] x, double y, double alpha) throws Exception {
		if(theta.length != x.length) throw new Exception("维数不一致！");
		int len = theta.length;
		double[] newTheta = new double[len];
		
		double dev = dev(theta, x, y);
		double gradient = 0;
		for(int i = 0; i < len; i++) {
			for(int j = 0; j < len; j++) {
				if (i == 0) {//theta_0的梯度计算稍微有点差别
					gradient += dev;
				} else {
					gradient += dev * x[j];
				}
			}
			newTheta[i] =  (theta[i] - alpha * (gradient / len));
		}
		return newTheta;
	}
	
	/**
	 * 梯度下降训练
	 * @param x_data
	 * @param y_data
	 * @param alpha
	 * @param deviation
	 * @param batchSize
	 * @param iterations
	 * @return
	 * @throws Exception
	 */
	public static double[] gdTrain(double[][] x_data, double[] y_data, double alpha, double deviation, int batchSize, int iterations) throws Exception {
		int parameter = x_data[0].length;
		int size = y_data.length;
		if (size % batchSize != 0) {
			throw new Exception("样本数必须是batchSize的整数倍！");
		}
		double[][] batchTheta = new double[batchSize][parameter];
		//随机初始化参数
		for(int j = 0; j < parameter; j++) {
			batchTheta[0][j] = Math.random();
		}
		for(int i = 1; i < batchSize; i++) {
			batchTheta[i] = batchTheta[0];
		}
//		for(int i = 0; i < batchSize; i++) {
//			for(int j = 0; j < parameter; j++) {
//				batchTheta[i][j] = Math.random();
//			}
//		}
		int count = 1;
		int batchCount = 1;
		int lossEqual = 0;
		double loss = 0;
		double lastLoss = 0;
		double[][] batchX = new double[batchSize][parameter];
		double[] batchY = new double[batchSize];
		double[] theta = new double[parameter];
		while (count <= iterations) {
			for(int i = 0; i < size; i = i + batchSize) {
				for(int j = 0; j < batchSize; j++) {
					batchX[j] = x_data[i + j];
					batchY[j] = y_data[i + j];
				}
				
				loss = loss(batchTheta, batchX, batchY);
				System.out.println("step = " + batchCount + "/" + count + "\tloss = " + loss);
				if(loss < deviation) {
					//System.out.println("训练完成！");
					break;
				}
				for(int j = 0; j < batchSize; j++) {
					theta = updateTheta(theta, batchX[j], batchY[j], alpha);
					batchTheta[j] = theta;
				}
				++batchCount;
			}
			//System.out.println("step = " + count + "\tloss = " + loss);
			alpha = alpha / 2;
			++count;
			batchCount = 1;
			if (lastLoss == loss) {
				++lossEqual;
			}
			lastLoss = loss;
			if(loss < deviation) {
				System.out.println("GD训练完成！");
				break;
			}
			if (lossEqual > 5) {
				System.out.println("loss值不在改变，GD训练完成！");
				break;
			}
		}
		for (int i = 0; i < theta.length; i++) {
			System.out.print(theta[i] + "\t");
		}
		System.out.println();
		return theta;
	}
	
	public static void main(String[] args) throws Exception {
		
		// 生成训练数据  y = a + b * x1 + c * x2
		int dataSize = 100;
		double[][] x_data = new double[dataSize][3];
		double[] y_data = new double[dataSize];
		double x1 = 0.0;
		double x2 = 0.0;
		for (int i = 0; i < dataSize; i++) {
			x_data[i][0] = 1.0;
			x1 = Math.random();
			x2 = Math.random();
			x_data[i][1] = x1;
			x_data[i][2] = x2;
			y_data[i] = 0.2 + 0.3 * x1 + 0.5 * x2;
			
		}
		
		// 生成训练数据 y = a + b * x1
		/*
		int dataSize = 100;
		double[][] x_data = new double[dataSize][2];
		double[] y_data = new double[dataSize];
		double x1 = 0.0;
		for (int i = 0; i < dataSize; i++) {
			x_data[i][0] = 1.0;
			x1 = Math.random();
			x_data[i][1] = x1;
			y_data[i] = 0.2 + 0.3 * x1;
			
		}
		*/
		
		//训练速率
		double alpha = 0.1; 
		double deviation = 0.00001;
		//每次更新参数使用样本大小
		int batchSize = 100;
		//迭代次数上限
		int iterations = 100000;
		
		//梯度下降法，如果batchSize的大小等于样本总数，则为批量梯度下降法
		gdTrain(x_data, y_data, alpha, deviation, batchSize, iterations);
	}

}
