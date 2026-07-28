package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
		 for (Game gameState : policyValues.keySet()) {
	        if (!gameState.isTerminal()) { //we ignore terminal states
	            
	        	List<Move> possible_m = gameState.getPossibleMoves();//to get all possible moves
	            Move randomMove = possible_m.get(new Random().nextInt(possible_m.size())); //generates a random move
	            curPolicy.put(gameState, randomMove); //sets the random move as the policy for the state
	        }
	    }
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		/* YOUR CODE HERE */
		double epsilon = 1e-6; //fixed value to check if the function has converged and this stop evaluation
	    boolean converged; //flag to track convergence

	    do {
	        converged = true; //initialize
	        HashMap<Game, Double> newValues = new HashMap<>(policyValues);//to store updated values

	        for (Game game_state:policyValues.keySet()) {
	            if (game_state.isTerminal()) {
	                newValues.put(game_state, 0.0); //terminal states have value 0
	                continue;
	            }

	            Move action = curPolicy.get(game_state); // Get action from the current policy
	            double newValue = 0.0;//to compute updated value of curr game state

	            for (TransitionProb transition : mdp.generateTransitions(game_state, action)) {
	                Game nextState = transition.outcome.sPrime;//value of next state
	                double reward = transition.outcome.localReward; //reward
	                double probability = transition.prob;//probability

	                newValue += probability * (reward + discount * policyValues.get(nextState)); //formula
	            }

	            // comapres and checks the diff in the value, if it is less than the epilson-> then it converged
	            if (Math.abs(newValues.get(game_state) - newValue) > epsilon) {
	                converged = false;
	            }

	            newValues.put(game_state, newValue);
	        }

	        policyValues = newValues; // Update values for the next iteration
	    } while (!converged);
		
		
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	
	// Helper function1-> to find the best move for a given state
	private Move getBestMove(Game state) {
	    Move bestMove = null; //initializing
	    double maxValue = -Double.MAX_VALUE;// initialize to start with the smallest value possible

	    for (Move move : state.getPossibleMoves()) {//loops though all possible moves
	        double currValue = 0.0;//initializing
	        for (TransitionProb tp : mdp.generateTransitions(state, move)) { //loop thorugh all possible transitions
	        	currValue += tp.prob * (tp.outcome.localReward + discount * policyValues.get(tp.outcome.sPrime));//sum all of it acc to the formula
	        }
	        if (currValue > maxValue) {//if current move has higher value, we update max value with the current value
	            maxValue = currValue;
	            bestMove = move;
	        }
	    }
	    return bestMove;//return the highest value
	}
	
	protected boolean improvePolicy()
	{
		/* YOUR CODE HERE */
		
		boolean policyStable = true;//setting it to true intially

	    for (Game state : curPolicy.keySet()) {//loop through all states
	        if (state.isTerminal()){//skip terminal states
				continue;
			} 

	        Move bestMove = getBestMove(state);//find best move
	        if (!bestMove.equals(curPolicy.get(state))) { //if best move is diff from current policy move
	            curPolicy.put(state, bestMove); // Update the policy with the best move.
	            policyStable = false;
	        }
	    }
	    return policyStable;//return whether policy is stable
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		/* YOUR CODE HERE */
		initRandomPolicy();//initialize a random policy
	    boolean policyStable = false; //to store boolean value
	    
	    while (!policyStable) {
	        evaluatePolicy(delta); //Evaluate the current policy
	        policyStable = improvePolicy();//improve the policy and check if its stable
	    } ;
	    this.policy = new Policy(curPolicy);//after convergence, we store resulting policy
		
		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
