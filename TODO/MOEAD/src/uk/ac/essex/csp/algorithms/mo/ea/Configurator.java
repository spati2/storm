package uk.ac.essex.csp.algorithms.mo.ea;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Properties;

/**
 * A Properties file based configuration.
 * 
 * @author wudong
 * 
 */
public class Configurator {

	public static final String Property_Do_Log = "do_log";
	public static final String Property_Do_Archive = "do_archive";
	public static final String Property_Random_Seed = "random_seed";

	// The total number of iteration set for the MOEAD running on GP.
	public static final String Property_Total_Iteration_Number = "total_generation";

	// The total number of evaluation, also is the stopping cretia of the
	// algorithms.
	public static final String Property_Total_Evaluation_Number = "total_evaluation";

	public static final String Property_Population_Size = "population_size";

	public int getPopSize() {
		return this.getIntegerProperty(Property_Population_Size);
	}

	public int getTotalGeneration() {
		if (this.hasKey(Property_Total_Iteration_Number))
			return this.getIntegerProperty(Property_Total_Iteration_Number);
		else
			return 0;
	}

	public void setTotalGeneration(int generation) {
		this.addProperty(Property_Total_Iteration_Number, Integer
				.toString(generation));
	}

	public int getTotalEvaluation() {
		if (this.hasKey(Property_Total_Evaluation_Number))
			return this.getIntegerProperty(Property_Total_Evaluation_Number);
		else
			return 0;
	}

	public void setTotalEvaluation(int evaluation) {
		this.addProperty(Property_Total_Evaluation_Number, Integer
				.toString(evaluation));
	}

	public long getRandomSeed() {
		return this.getLongProperty(Property_Random_Seed);
	}

	public void setRandomSeed(long randomseed) {
		this.addProperty(Property_Random_Seed, Long.toString(randomseed));
	}

	public void setDoLog(boolean log) {
		this.addProperty(Property_Do_Log, Boolean.toString(log));
	}

	public boolean doLog() {
		return this.getBoolProperty(Property_Do_Log);
	}

	public void setDoArchive(boolean archive) {
		this.addProperty(Property_Do_Archive, Boolean.toString(archive));
	}

	public boolean doArchive() {
		return this.getBoolProperty(Property_Do_Archive);
	}

	private Properties properties = new Properties();

	/**
	 * This method is load the basic configuration from the file.
	 * 
	 * @param filename
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void loadDefaultProperties(String filename)
			throws FileNotFoundException, IOException {
		properties.clear();
		properties.load(new FileInputStream(filename));
	}

	/**
	 * This method is to load additional basic configuration from the file and
	 * it will override existed properties.
	 * 
	 * @param filename
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void addProperties(String filename) throws FileNotFoundException,
			IOException {
		Properties pp = new Properties();
		pp.load(new FileInputStream(filename));
		properties.putAll(pp);
	}

	/**
	 * Add additional properties on the fly.
	 * 
	 * @param key
	 * @param value
	 */
	public void addProperty(String key, String value) {
		properties.setProperty(key, value);
	}

	protected int getIntegerProperty(String key) {
		String object = this.properties.getProperty(key);
		return Integer.parseInt(object);
	}

	protected double getDoubleProperty(String key) {
		String object = this.properties.getProperty(key);
		return Double.parseDouble(object);
	}

	protected boolean getBoolProperty(String key) {
		String object = this.properties.getProperty(key);
		return Boolean.parseBoolean(object);
	}

	protected String getStringProperty(String key) {
		return this.properties.getProperty(key);
	}

	protected long getLongProperty(String key) {
		String object = this.properties.getProperty(key);
		return Long.parseLong(object);
	}

	protected boolean hasKey(String key) {
		return this.properties.getProperty(key) != null;
	}

	/**
	 * List the properties keys and values for examine.
	 * 
	 * @param out
	 */
	public void listProperties(PrintStream out) {
		this.properties.list(out);
	}

}
