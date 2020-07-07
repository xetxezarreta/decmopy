/**
 * DECMO2_VerCZ.java
 *
 * @author Ciprian Zavoianu
 * @version 1.0
 */
package jmetal.metaheuristics.DECMO2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import jmetal.base.Algorithm;
import jmetal.base.Operator;
import jmetal.base.Problem;
import jmetal.base.Solution;
import jmetal.base.SolutionSet;
import jmetal.base.operator.crossover.CrossoverFactory;
import jmetal.base.operator.mutation.MutationFactory;
import jmetal.base.operator.selection.SelectionFactory;
import jmetal.problems.Kursawe;
import jmetal.problems.DTLZ.DTLZ1;
import jmetal.problems.DTLZ.DTLZ2;
import jmetal.problems.DTLZ.DTLZ3;
import jmetal.problems.DTLZ.DTLZ4;
import jmetal.problems.DTLZ.DTLZ5;
import jmetal.problems.DTLZ.DTLZ6;
import jmetal.problems.DTLZ.DTLZ7;
import jmetal.problems.LZ09.LZ09_F1;
import jmetal.problems.LZ09.LZ09_F2;
import jmetal.problems.LZ09.LZ09_F3;
import jmetal.problems.LZ09.LZ09_F4;
import jmetal.problems.LZ09.LZ09_F5;
import jmetal.problems.LZ09.LZ09_F6;
import jmetal.problems.LZ09.LZ09_F7;
import jmetal.problems.LZ09.LZ09_F8;
import jmetal.problems.LZ09.LZ09_F9;
import jmetal.problems.WFG.WFG1;
import jmetal.problems.WFG.WFG2;
import jmetal.problems.WFG.WFG3;
import jmetal.problems.WFG.WFG4;
import jmetal.problems.WFG.WFG5;
import jmetal.problems.WFG.WFG6;
import jmetal.problems.WFG.WFG7;
import jmetal.problems.WFG.WFG8;
import jmetal.problems.WFG.WFG9;
import jmetal.problems.ZDT.ZDT1;
import jmetal.problems.ZDT.ZDT2;
import jmetal.problems.ZDT.ZDT3;
import jmetal.problems.ZDT.ZDT4;
import jmetal.problems.ZDT.ZDT6;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.Spea2Fitness;

/**
 * This class implements the GDE3 algorithm.
 */
public class DECMO2_VerCZ extends Algorithm {

	final int MIN_VALUES = 0;
	final int MAX_VALUES = 1;

	List<List<Double>> extremeValues = new ArrayList<List<Double>>();

	String dataDirectory;

	private final Problem problem_;

	private static String PROBLEM_NAME = "Problem";

	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public DECMO2_VerCZ(Problem problem) {
		this.problem_ = problem;

		List<Double> minValues = new ArrayList<Double>();
		List<Double> maxValues = new ArrayList<Double>();
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			minValues.add(Double.MAX_VALUE);
			maxValues.add(Double.MIN_VALUE);
		}
		extremeValues.add(minValues);
		extremeValues.add(maxValues);
	} // DECMO2

	private void clearNfeHistory(List<DirectionRec> directionalArchive) {

		for (DirectionRec dr : directionalArchive) {
			dr.setNfeSinceLastUpdate(0);
		}

	}

	private double computeEuclideanDistance(double[] vector1, double[] vector2) {
		double value = 0;
		for (int i = 0; i < vector1.length; i++) {
			value += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
		}
		return Math.sqrt(value);
	}

	private int[] computeNeighbourhoodNfeSinceLastUpdate(List<List<Integer>> neighbourhoods,
			List<DirectionRec> directionalArchive, int intensificationClusters) {

		List<CompRec> averageNfe = new ArrayList<CompRec>();
		int ID = 0;
		for (List<Integer> neighbourhood : neighbourhoods) {
			double avg = 0;
			for (Integer nID : neighbourhood) {
				avg += directionalArchive.get(nID).getNfeSinceLastUpdate();
			}
			avg /= neighbourhood.size();

			averageNfe.add(new CompRec(ID, avg));
			ID++;
		}

		Collections.sort(averageNfe);

		int[] result = new int[intensificationClusters];
		for (int i = 0; i < intensificationClusters; i++) {
			result[i] = averageNfe.get(averageNfe.size() - 1 - i).getID();
		}
		return result;
	}

	private List<DirectionRec> createDirectionalArchive(double[][] lambda) {

		List<DirectionRec> directionalArchive = new ArrayList<DirectionRec>();

		for (int i = 0; i < lambda.length; i++) {
			DirectionRec di = new DirectionRec(i, lambda[i], null, Double.MAX_VALUE, 0);
			directionalArchive.add(di);
		}
		return directionalArchive;
	}

	private List<List<Integer>> createNeighbourhoods(List<DirectionRec> dirArchive, int neighborhoodSize) {
		List<List<Integer>> neighbourhoods = new ArrayList<List<Integer>>();

		for (DirectionRec di1 : dirArchive) {
			List<CompRec> distToNeighbour = new ArrayList<CompRec>();
			for (DirectionRec di2 : dirArchive) {
				if (di1.getID() != di2.getID()) {
					distToNeighbour.add(new CompRec(di2.getID(),
							computeEuclideanDistance(di1.getWeightVector(), di2.getWeightVector())));
				}
			}

			Collections.sort(distToNeighbour);
			List<Integer> neighbourhood = new ArrayList<Integer>();
			for (int i = 0; i < neighborhoodSize; i++) {
				if (i < distToNeighbour.size()) {
					neighbourhood.add(distToNeighbour.get(i).getID());
				}
			}
			neighbourhoods.add(neighbourhood);
		}

		return neighbourhoods;
	}

	/**
	 * initUniformWeights
	 */
	private double[][] createUniformWeights(int dirArchiveSize, int nrOfObjectives) {

		double[][] lambda = new double[dirArchiveSize][nrOfObjectives];

		if ((nrOfObjectives == 2) && (dirArchiveSize < 500)) {
			for (int n = 0; n < dirArchiveSize; n++) {
				double a = 1.0 * n / (dirArchiveSize - 1);
				lambda[n][0] = a;
				lambda[n][1] = 1 - a;
				System.out.println(lambda[n][0]);
				System.out.println(lambda[n][1]);
			} // for
		} // if
		else {
			String dataFileName;
			dataFileName = "W" + nrOfObjectives + "D_" + dirArchiveSize + ".dat";

			System.out.println(dataDirectory);
			System.out.println(dataDirectory + "/" + dataFileName);

			DistribGen dg = new DistribGen();
			dg.createDistribution(problem_.getNumberOfObjectives(), dirArchiveSize, dataDirectory + "/" + dataFileName);

			try {
				// Open the file
				FileInputStream fis = new FileInputStream(dataDirectory + "/" + dataFileName);
				InputStreamReader isr = new InputStreamReader(fis);
				BufferedReader br = new BufferedReader(isr);

				// int numberOfObjectives = 0;
				int i = 0;
				int j = 0;
				String aux = br.readLine();
				while (aux != null) {
					StringTokenizer st = new StringTokenizer(aux);
					j = 0;
					// numberOfObjectives = st.countTokens();
					while (st.hasMoreTokens()) {
						double value = (new Double(st.nextToken())).doubleValue();
						lambda[i][j] = value;
						// System.out.println("lambda["+i+","+j+"] = " + value)
						// ;
						j++;
					}
					aux = br.readLine();
					i++;
				}
				br.close();
			} catch (Exception e) {
				System.out.println(
						"initUniformWeight: failed when reading for file: " + dataDirectory + "/" + dataFileName);
				e.printStackTrace();
			}
		} // else

		return lambda;
	} // initUniformWeights

	/**
	 * evaluateTchebycheffFitness(Solution individual, double[] weightVector)
	 */
	private double evaluateTchebycheffFitness(Solution individual, double[] lambda) {
		double max = Double.MIN_VALUE;

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			double diff = Math.abs(individual.getObjective(i) - extremeValues.get(MIN_VALUES).get(i));

			double tcheFuncVal;
			if (lambda[i] == 0) {
				tcheFuncVal = 0.000001 * diff;
			} else {
				tcheFuncVal = diff * lambda[i];
			}
			if (tcheFuncVal > max) {
				max = tcheFuncVal;
			}
		} // for

		return max;
	} // evaluateTchebycheffFitness

	/**
	 * Runs the DECMO2 algorithm.
	 * 
	 * @return a <code>SolutionSet</code> that is a set of non dominated
	 *         solutions as a result of the algorithm execution
	 * @throws JMException
	 */
	@Override
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		SolutionSet pool1;
		SolutionSet pool2;
		SolutionSet poolA;

		SolutionSet offspringPop1;
		SolutionSet offspringPop2;
		SolutionSet offspringPop3;

		SolutionSet dirInsertPool1;
		SolutionSet dirInsertPool2;
		SolutionSet dirInsertPool3;

		Comparator dominance = new jmetal.base.operator.comparator.DominanceComparator();

		/** Selection operator 1 */
		Operator selectionOperator1 = SelectionFactory.getSelectionOperator("BinaryTournament");

		/** Selection operator 2 */
		Operator selectionOperator2 = SelectionFactory.getSelectionOperator("DifferentialEvolutionSelection");

		/** Crossover operator 1 */
		Operator crossoverOperator1 = CrossoverFactory.getCrossoverOperator("SBXCrossover");
		crossoverOperator1.setParameter("probability", 1.0);
		crossoverOperator1.setParameter("distributionIndex", 20.0);

		/** Crossover operator 2 */
		Operator crossoverOperator2 = CrossoverFactory.getCrossoverOperator("DifferentialEvolutionCrossover");
		crossoverOperator2.setParameter("CR", 0.2);
		crossoverOperator2.setParameter("F", 0.5);

		/** Crossover operator 3 */
		Operator crossoverOperator3 = CrossoverFactory.getCrossoverOperator("DifferentialEvolutionCrossover");
		crossoverOperator3.setParameter("CR", 1.0);
		crossoverOperator3.setParameter("F", 0.5);

		/** Mutation operator 1 */
		Operator mutationOperator1 = MutationFactory.getMutationOperator("PolynomialMutation");
		mutationOperator1.setParameter("probability", 1.0 / problem_.getNumberOfVariables());
		mutationOperator1.setParameter("distributionIndex", 20.0);

		/** Array that stores the "generational" HV quality */
		List<Double> generationalHV = new ArrayList<Double>();

		/** Main run parameters */
		int populationSize = ((Integer) this.getInputParameter("individualPopulationSize")).intValue();
		int reportInterval = ((Integer) getInputParameter("reportInterval")).intValue();
		int maxEvaluations = ((Integer) this.getInputParameter("maxEvaluations")).intValue();

		dataDirectory = this.getInputParameter("dataDirectory").toString();

		if (problem_ instanceof DTLZ1) {
			PROBLEM_NAME = "DTLZ1_7";
		}
		if (problem_ instanceof DTLZ2) {
			PROBLEM_NAME = "DTLZ2_12";
		}
		if (problem_ instanceof DTLZ3) {
			PROBLEM_NAME = "DTLZ3_12";
		}
		if (problem_ instanceof DTLZ4) {
			PROBLEM_NAME = "DTLZ4_12";
		}
		if (problem_ instanceof DTLZ5) {
			PROBLEM_NAME = "DTLZ5_12";
		}
		if (problem_ instanceof DTLZ6) {
			PROBLEM_NAME = "DTLZ6_12";
		}
		if (problem_ instanceof DTLZ7) {
			PROBLEM_NAME = "DTLZ7_22";
		}
		if (problem_ instanceof ZDT6) {
			PROBLEM_NAME = "ZDT6_10";
		}
		if (problem_ instanceof ZDT4) {
			PROBLEM_NAME = "ZDT4_10";
		}
		if (problem_ instanceof ZDT3) {
			PROBLEM_NAME = "ZDT3_10";
		}
		if (problem_ instanceof ZDT2) {
			PROBLEM_NAME = "ZDT2_30";
		}
		if (problem_ instanceof ZDT1) {
			PROBLEM_NAME = "ZDT1_30";
		}
		if (problem_ instanceof WFG1) {
			PROBLEM_NAME = "WFG1_6";
		}
		if (problem_ instanceof WFG2) {
			PROBLEM_NAME = "WFG2_6";
		}
		if (problem_ instanceof WFG3) {
			PROBLEM_NAME = "WFG3_6";
		}
		if (problem_ instanceof WFG4) {
			PROBLEM_NAME = "WFG4_6";
		}
		if (problem_ instanceof WFG5) {
			PROBLEM_NAME = "WFG5_6";
		}
		if (problem_ instanceof WFG6) {
			PROBLEM_NAME = "WFG6_6";
		}
		if (problem_ instanceof WFG7) {
			PROBLEM_NAME = "WFG7_6";
		}
		if (problem_ instanceof WFG8) {
			PROBLEM_NAME = "WFG8_6";
		}
		if (problem_ instanceof WFG9) {
			PROBLEM_NAME = "WFG9_6";
		}
		if (problem_ instanceof LZ09_F1) {
			PROBLEM_NAME = "LZ09_F1_30";
		}
		if (problem_ instanceof LZ09_F2) {
			PROBLEM_NAME = "LZ09_F2_30";
		}
		if (problem_ instanceof LZ09_F3) {
			PROBLEM_NAME = "LZ09_F3_30";
		}
		if (problem_ instanceof LZ09_F4) {
			PROBLEM_NAME = "LZ09_F4_30";
		}
		if (problem_ instanceof LZ09_F5) {
			PROBLEM_NAME = "LZ09_F5_30";
		}
		if (problem_ instanceof LZ09_F6) {
			PROBLEM_NAME = "LZ09_F6_30";
		}
		if (problem_ instanceof LZ09_F7) {
			PROBLEM_NAME = "LZ09_F7_10";
		}
		if (problem_ instanceof LZ09_F8) {
			PROBLEM_NAME = "LZ09_F8_10";
		}
		if (problem_ instanceof LZ09_F9) {
			PROBLEM_NAME = "LZ09_F9_30";
		}
		if (problem_ instanceof Kursawe) {
			PROBLEM_NAME = "KSW_10";
		}

		/** Initializations */
		Solution parent1[] = new Solution[2];
		Solution parent2[];
		Solution parent3[];

		// size of elite subset used for fitness sharing between subpopulations
		int nrOfDirectionalSolutionsToEvolve = populationSize / 5;
		// Initialize the main subpopulation variables
		// subpopulation 1
		int pool1Size = populationSize - (nrOfDirectionalSolutionsToEvolve / 2);
		// seubpopulation 2
		int pool2Size = populationSize - (nrOfDirectionalSolutionsToEvolve / 2);
		int mixInterval = 1;

		System.out.println(pool1Size + " - " + nrOfDirectionalSolutionsToEvolve + " - " + mixInterval);

		// Initialize some local and global variables
		pool1 = new SolutionSet(pool1Size);
		pool2 = new SolutionSet(pool2Size);
		int evaluations = 0;
		int currentGen = 0;
		int directionalArchiveSize = 2 * populationSize;
		double[][] weights = createUniformWeights(directionalArchiveSize, problem_.getNumberOfObjectives());

		List<DirectionRec> directionalArchive = createDirectionalArchive(weights);
		List<List<Integer>> neighbourhoods = createNeighbourhoods(directionalArchive, populationSize);
		int nrOfReplacements = 1;

		int iniID = 0;

		/** Create the initial pools */
		// pool1
		Solution newSolution;
		for (int i = 0; i < pool1Size; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			pool1.add(newSolution);

			updateExtremeValues(newSolution);
			DirectionRec dr = directionalArchive.get(iniID);
			dr.setCurrSol(newSolution);
			iniID++;
		}
		// pool2
		for (int i = 0; i < pool2Size; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			pool2.add(newSolution);

			updateExtremeValues(newSolution);
			DirectionRec dr = directionalArchive.get(iniID);
			dr.setCurrSol(newSolution);
			iniID++;
		}
		// directional archive initialization
		poolA = new SolutionSet(directionalArchiveSize);
		while (iniID < directionalArchiveSize) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			poolA.add(newSolution);

			updateExtremeValues(newSolution);
			DirectionRec dr = directionalArchive.get(iniID);
			dr.setCurrSol(newSolution);
			iniID++;
		}

		int mix = mixInterval;

		QualityIndicator indicator = new QualityIndicator(problem_,
				"data\\input\\trueParetoFronts\\" + PROBLEM_NAME + ".pareto");

		Random rndGen = new Random(System.currentTimeMillis());

		double[] insertionRate = new double[3];
		insertionRate[0] = 0;
		insertionRate[1] = 0;
		insertionRate[2] = 0;
		int[] bonusEvals = new int[3];
		bonusEvals[0] = 0;
		bonusEvals[1] = 0;
		bonusEvals[2] = nrOfDirectionalSolutionsToEvolve;
		boolean testRun = true;

		/** record the generational HV of the initial population */
		SolutionSet combiAll = new SolutionSet();
		int cGen = evaluations / reportInterval;
		if (cGen > 0) {

			combiAll = (((combiAll.union(pool1)).union(pool2)).union(poolA));
			Spea2Fitness spea0 = new Spea2Fitness(combiAll);
			spea0.fitnessAssign();
			combiAll = spea0.environmentalSelection(pool1Size + pool2Size);
			double hVal = indicator.getHypervolume(combiAll);
			System.out.println("Hypervolume: " + cGen + " - " + hVal);
			for (int j = 0; j < cGen; j++) {
				generationalHV.add(hVal);
			}
			currentGen = cGen;
		}

		/** The main loop of the algorithm */
		while (evaluations < maxEvaluations) {

			offspringPop1 = new SolutionSet(pool1Size * 4);
			offspringPop2 = new SolutionSet(pool2Size * 4);
			offspringPop3 = new SolutionSet(directionalArchiveSize);

			dirInsertPool1 = new SolutionSet(pool1Size * 2);
			dirInsertPool2 = new SolutionSet(pool1Size * 2);
			dirInsertPool3 = new SolutionSet(pool1Size * 2);

			/** evolve pool1 - using SPEA2 evolutionary model */

			// int improvements = 0;
			int nfe = 0;
			while (nfe < pool1Size + bonusEvals[0]) {

				parent1[0] = (Solution) selectionOperator1.execute(pool1);
				parent1[1] = (Solution) selectionOperator1.execute(pool1);

				Solution[] children = (Solution[]) crossoverOperator1.execute(parent1);
				Solution child1a = children[0];
				mutationOperator1.execute(child1a);

				problem_.evaluate(child1a);
				problem_.evaluateConstraints(child1a);
				evaluations++;
				nfe++;

				offspringPop1.add(child1a);
				dirInsertPool1.add(child1a);
			} // while pool1

			/** evolve pool2 - using DEMO SP evolutionary model */
			int i = 0;
			List<Integer> unselectedIDs = new ArrayList<Integer>();
			for (int ID = 0; ID < pool2.size(); ID++) {
				unselectedIDs.add(ID);
			}

			nfe = 0;

			while (nfe < pool2Size + bonusEvals[1]) {

				int index = rndGen.nextInt(unselectedIDs.size());
				i = unselectedIDs.get(index);
				unselectedIDs.remove(index);

				parent2 = (Solution[]) selectionOperator2.execute(new Object[] { pool2, i });

				Solution child2 = (Solution) crossoverOperator2.execute(new Object[] { pool2.get(i), parent2 });

				problem_.evaluate(child2);
				problem_.evaluateConstraints(child2);
				evaluations++;
				nfe++;

				int result;
				result = dominance.compare(pool2.get(i), child2);
				if (result == -1) { // Solution i dominates child
					offspringPop2.add(pool2.get(i));
				} // if
				else if (result == 1) { // child dominates
					offspringPop2.add(child2);
				} // else if
				else { // the two solutions are non-dominated
					offspringPop2.add(child2);
					offspringPop2.add(pool2.get(i));
				} // else

				dirInsertPool2.add(child2);

				if (unselectedIDs.size() == 0) {
					for (int ID = 0; ID < pool2.size(); ID++) {
						unselectedIDs.add(PseudoRandom.randInt(0, pool2.size() - 1));
					}
				}
			} // while pool2

			/** evolve pool3 - Directional Decomposition DE/rand/1/bin */
			int[] IDs = computeNeighbourhoodNfeSinceLastUpdate(neighbourhoods, directionalArchive,
					nrOfDirectionalSolutionsToEvolve);

			nfe = 0;
			for (int j = 0; j < IDs.length; j++) {

				if (nfe < bonusEvals[2]) {
					nfe++;
				} else {
					break;
				}

				int cID = IDs[j];
				Solution chosenSol;
				if (directionalArchive.get(cID).getCurrSol() != null) {
					chosenSol = directionalArchive.get(cID).getCurrSol();
				} else {
					chosenSol = pool1.get(0);
					System.out.println("error!");
				}

				parent3 = new Solution[3];

				int r1;
				int r2;
				int r3;

				r1 = (PseudoRandom.randInt(0, neighbourhoods.get(cID).size() - 1));
				do {
					r2 = (PseudoRandom.randInt(0, neighbourhoods.get(cID).size() - 1));
				} while (r2 == r1);
				do {
					r3 = (PseudoRandom.randInt(0, neighbourhoods.get(cID).size() - 1));
				} while (r3 == r1 || r3 == r2);

				parent3[0] = directionalArchive.get(r1).getCurrSol();
				parent3[1] = directionalArchive.get(r2).getCurrSol();
				parent3[2] = directionalArchive.get(r3).getCurrSol();

				Solution child3 = (Solution) crossoverOperator3.execute(new Object[] { chosenSol, parent3 });
				mutationOperator1.execute(child3);

				problem_.evaluate(child3);
				problem_.evaluateConstraints(child3);
				evaluations++;

				dirInsertPool3.add(child3);
			} // for pool3

			/** Compute directional improvements */
			// poo1
			int improvements = 0;
			for (int j = 0; j < dirInsertPool1.size(); j++) {
				Solution testSol = dirInsertPool1.get(j);

				updateExtremeValues(testSol);
				improvements += updateNeighbourhoods(directionalArchive, testSol, nrOfReplacements);
			}
			insertionRate[0] += (1.0 * improvements) / dirInsertPool1.size();
			// pool2
			improvements = 0;
			for (int j = 0; j < dirInsertPool2.size(); j++) {
				Solution testSol = dirInsertPool2.get(j);

				updateExtremeValues(testSol);
				improvements += updateNeighbourhoods(directionalArchive, testSol, nrOfReplacements);
			}
			insertionRate[1] += (1.0 * improvements) / dirInsertPool2.size();
			// pool3
			improvements = 0;
			for (int j = 0; j < dirInsertPool3.size(); j++) {
				Solution testSol = dirInsertPool3.get(j);

				updateExtremeValues(testSol);
				improvements += updateNeighbourhoods(directionalArchive, testSol, nrOfReplacements);
			}
			insertionRate[2] += (1.0 * improvements) / dirInsertPool3.size();

			for (DirectionRec dr : directionalArchive) {
				offspringPop3.add(dr.getCurrSol());
			}

			offspringPop1 = offspringPop1.union(pool1);
			Spea2Fitness spea1 = new Spea2Fitness(offspringPop1);
			spea1.fitnessAssign();
			pool1 = spea1.environmentalSelection(pool1Size);

			Spea2Fitness spea2 = new Spea2Fitness(offspringPop2);
			spea2.fitnessAssign();
			pool2 = spea2.environmentalSelection(pool2Size);

			SolutionSet combi = new SolutionSet();
			mix--;
			if (mix == 0) {
				mix = mixInterval;
				combi = ((combi.union(pool1)).union(pool2)).union(offspringPop3);
				System.out.println("Combi size: " + combi.size());
				Spea2Fitness spea5 = new Spea2Fitness(combi);
				spea5.fitnessAssign();
				combi = spea5.environmentalSelection(nrOfDirectionalSolutionsToEvolve);

				insertionRate[0] /= mixInterval;
				insertionRate[1] /= mixInterval;
				insertionRate[2] /= mixInterval;
				System.out.println("Insertion rates: " + insertionRate[0] + " - " + insertionRate[1] + " - "
						+ insertionRate[2] + " - Test run:" + testRun);

				if (testRun) {
					if ((insertionRate[0] > insertionRate[1]) && (insertionRate[0] > insertionRate[2])) {
						System.out.println("SPEA2 win - bonus run!");
						bonusEvals[0] = nrOfDirectionalSolutionsToEvolve;
						bonusEvals[1] = 0;
						bonusEvals[2] = 0;
					}
					if ((insertionRate[1] > insertionRate[0]) && (insertionRate[1] > insertionRate[2])) {
						System.out.println("DE win - bonus run!");
						bonusEvals[0] = 0;
						bonusEvals[1] = nrOfDirectionalSolutionsToEvolve;
						bonusEvals[2] = 0;
					}
					if ((insertionRate[2] > insertionRate[0]) && (insertionRate[2] > insertionRate[1])) {
						System.out.println("Directional win - no bonus!");
						bonusEvals[0] = 0;
						bonusEvals[1] = 0;
						bonusEvals[2] = nrOfDirectionalSolutionsToEvolve;
					}
				} else {
					System.out.println("Test run - no bonus!");
					bonusEvals[0] = 0;
					bonusEvals[1] = 0;
					bonusEvals[2] = nrOfDirectionalSolutionsToEvolve;
				}
				testRun = !testRun;

				insertionRate[0] = 0.0;
				insertionRate[1] = 0.0;
				insertionRate[2] = 0.0;

				pool1 = pool1.union(combi);
				pool2 = pool2.union(combi);

				System.out.println("Sizes: " + pool1.size() + " " + pool2.size());

				spea1 = new Spea2Fitness(pool1);
				spea1.fitnessAssign();
				pool1 = spea1.environmentalSelection(pool1Size);

				spea2 = new Spea2Fitness(pool2);
				spea2.fitnessAssign();
				pool2 = spea2.environmentalSelection(pool2Size);

				clearNfeHistory(directionalArchive);
			}

			double hVal1 = indicator.getHypervolume(pool1);
			double hVal2 = indicator.getHypervolume(pool2);
			double hVal3 = indicator.getHypervolume(offspringPop3);

			int newGen = evaluations / reportInterval;
			if (newGen > currentGen) {

				System.out.println("Hypervolume: " + newGen + " - " + hVal1 + " - " + hVal2 + " - " + hVal3);

				combi = (((combi.union(pool1)).union(pool2)).union(offspringPop3));
				Spea2Fitness spea5 = new Spea2Fitness(combi);
				spea5.fitnessAssign();
				combi = spea5.environmentalSelection(populationSize * 2);
				double hVal = indicator.getHypervolume(combi);
				for (int j = currentGen; j < newGen; j++) {
					generationalHV.add(hVal);
				}
				currentGen = newGen;
			}

		} // while (main evolutionary cycle)

		// write runtime generational HV to file
		String sGenHV = "";
		for (Double d : generationalHV) {
			sGenHV += d + ",";
		}

		try {
			File hvFile = new File("data\\output\\runtimePerformance\\DECMO2\\SolutionSetSize" + 2 * populationSize
					+ "\\" + PROBLEM_NAME + "\\HV.csv");
			File dir = new File(hvFile.getParent());
			if (!dir.exists() && !dir.mkdirs()) {
				System.out.println("Could not create directory path: ");
			}
			if (!hvFile.exists()) {
				hvFile.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(hvFile, true));
			bw.write(sGenHV + "\n");
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		/**
		 * Return the final combined non-dominated set of maximum size =
		 * (populationSize * 2)
		 */
		combiAll = new SolutionSet();
		combiAll = (((combiAll.union(pool1)).union(pool2)).union(poolA));
		Spea2Fitness speaF = new Spea2Fitness(combiAll);
		speaF.fitnessAssign();
		combiAll = speaF.environmentalSelection(populationSize * 2);
		return combiAll;
	} // execute

	/**
	 * updateExtremeValues(Solution currentSolution)
	 */
	private void updateExtremeValues(Solution sol) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			double objValue = sol.getObjective(i);
			if (objValue < extremeValues.get(MIN_VALUES).get(i)) {
				extremeValues.get(MIN_VALUES).set(i, objValue);
			}
			if (objValue > extremeValues.get(MAX_VALUES).get(i)) {
				extremeValues.get(MAX_VALUES).set(i, objValue);
			}
		}
	}

	private int updateNeighbourhoods(List<DirectionRec> directionalArchive, Solution newSolution,
			int nrOfReplacements) {

		List<CompRec> improvedDistances = new ArrayList<CompRec>();
		boolean isImprovement = false;

		for (DirectionRec cdr : directionalArchive) {
			double newFitnessValue = evaluateTchebycheffFitness(newSolution, cdr.getWeightVector());
			if (newFitnessValue < cdr.getFitnessValue()) {
				improvedDistances.add(new CompRec(cdr.getID(), newFitnessValue));
				isImprovement = true;
			} else {
				cdr.setNfeSinceLastUpdate(cdr.getNfeSinceLastUpdate() + 1);
			}
		}

		Collections.sort(improvedDistances);
		Collections.reverse(improvedDistances);

		if (isImprovement) {
			for (int i = 0; i < nrOfReplacements; i++) {

				// int j = (new Random()).nextInt(improvedDistances.size());
				// int j = improvedDistances.size() - 1;
				int j = 0;

				DirectionRec cdr = directionalArchive.get(improvedDistances.get(j).getID());
				cdr.setCurrSol(newSolution);
				cdr.setFitnessValue(improvedDistances.get(j).getValue());
				cdr.setNfeSinceLastUpdate(0);
			}
			return 1;
		}

		return 0;
	}

} // DECMO2_VerCZ