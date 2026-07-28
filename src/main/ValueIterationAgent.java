package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		/* YOUR CODE HERE
		 */
		for (int i = 0; i < k; i++) { // we run the loop till V(k+1) according to the formula of value iteration
	        Map<Game, Double> updated_value = new HashMap<>(); //stores updated values after each iterations

	        for (Game game_state : valueFunction.keySet()) { //loops through all game states
	            if (game_state.isTerminal()) {//if game state is terminal (ends because of win,loss or draw)
	            	updated_value.put(game_state, valueFunction.get(game_state)); //insert the same value since no action is taken from it
	                continue; //skips to the next game state
	            }

	            //bellman equation: max(sum of(probability*[reward + discount * value of next function]))
	            double max_value = Double.NEGATIVE_INFINITY; //to store max value among all actions
	            for (Move move : game_state.getPossibleMoves()) { //loop to go through all actions it can make for ccurrent state
	                double value = 0.0;
	                List<TransitionProb> transitions = mdp.generateTransitions(game_state, move); //using generate transitions to retrieve possible outcomes using tttmdp
	                
	                //implementing sum of(probability *[reward + discount * value of next function]) part of the formula
	                for (TransitionProb transition : transitions) {
	                    Game nextGame = transition.outcome.sPrime;//value of next state
	                    double reward = transition.outcome.localReward; //reward
	                    double probability = transition.prob; //probability
	                    
	                    value += probability * (reward + discount * valueFunction.get(nextGame)); //update the expected value with the formula 
	                }
	                max_value = Math.max(max_value,value);//taking the max value
	            }
	            updated_value.put(game_state,max_value);//stores the optimal value
	        }
	        valueFunction = updated_value;//replaces old value after completing all states
		}        
	}
	
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
		Policy policy = new Policy(); //initialize an empty policy using policy object instead of hashmap

	    for (Game game_state : valueFunction.keySet()) {//iterate over all game state
	        if (game_state.isTerminal()) {//skip terminal sates as no action is needed
	            continue;
	        }

	        Move bestMove = null; //to store move w highest value
	        double maxExpectedValue = Double.NEGATIVE_INFINITY;

	        for (Move move : game_state.getPossibleMoves()) {//iterate all possible actions
	            double expectedValue = 0.0;
	            
	            List<TransitionProb> transitions = mdp.generateTransitions(game_state, move);
	            for (TransitionProb transition : transitions) {
	                Game nextGame = transition.outcome.sPrime; //value of next game state
	                double reward = transition.outcome.localReward;//reward
	                double probability = transition.prob;//probabilty

	                //update value using formula
	                expectedValue += probability * (reward + discount * valueFunction.get(nextGame));
	            }

	            //update best move if expected value of current sate is higher
	            if (expectedValue > maxExpectedValue) {
	                maxExpectedValue = expectedValue;
	                bestMove = move;
	            }
	        }

	        if (bestMove != null) {//if best move is found updates policy with optimal move
	            policy.policy.put(game_state, bestMove);
	        }
	    }

	    return policy;//return extracted policy after all game state
	}

	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
