// Figura 3.10:GradeBook.java
// Classe GradeBook com um construtor para inicializar o
// nome de um curso
public class GradeBook
{
	private String courseName; // nome do curso para esse GradeBook

	// o contrutor inicializa courseName com o argumento String
	public GradeBook( String name )
	{
		courseName = name; // inicializa courseName
	} // fim do construtor

	// metodo para configurar o nome do curso
	public void setCourseName( String name )
	{
		courseName = name; // armazena o nome do curso
	} // fim do metodo setCourseName

	// metodo para recuperar o nome do curso
	public String getCourseName()
	{
		return courseName;
	} // fim do metodo getCourseName

	// exibe uma mensagem de boas-vindas para o usuario GradeBook
	public void displayMessage()
	{
		// essa instrucao chama o metodo getCourseName para obter o
		// nome do curso que esse GradeBook representa
		System.out.printf( "Welcome to the GradeBook for \n%s!\n",
			getCourseName() );
	} // fim do metodo displayMessage
} // fim da classe GradeBook